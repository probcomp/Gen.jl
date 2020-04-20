using Gen
using PyPlot
using Random: seed!
using ForwardDiff

include("dirichlet.jl")

#########
# model #
#########

struct ClusterParams{T}
    mu_x::T
    mu_y::T
    var_x::T
    var_y::T
end

@gen function model(n:Int, k::Int)
    # n is number of data points
    # k is the number of clusters

    # prior on the mixture weights
    mixture_weights ~ dirichlet(4 * ones(k))

    # sample cluster parameters
    params = Vector{ClusterParams{Float64}}(undef, k)
    for i=1:k
        mu_x = ({(:mu_x, i)} ~ normal(0, 10))
        mu_y = ({(:mu_y, i)} ~ normal(0, 10))
        var_x = ({(:var_x, i)} ~ inv_gamma(1, 1)) # was 0.1
        var_y = ({(:var_y, i)} ~ inv_gamma(1, 1)) # was 0.1
        params[i] = ClusterParams(mu_x, mu_y, var_x, var_y)
    end

    # sample assignments and data
    for i=1:n
        z = ({(:z, i)} ~ categorical(mixture_weights))
        {(:x, i)} ~ normal(params[z].mu_x, sqrt(max(1e-10, params[z].var_x)))
        {(:y, i)} ~ normal(params[z].mu_y, sqrt(max(1e-10, params[z].var_y)))
    end
    return params
end

################################
# MCMC move on mixture weights #
################################

@gen function mixture_weight_1d_proposal(trace)
    n, k = get_args(trace)
    # pick two clusters whose weight should change
    i ~ uniform_discrete(1, k)
    j ~ uniform_discrete(1, k)
    if i != j
        u ~ uniform_continuous(0, 1)
    end
    return nothing
end

# TODO make these tree arguments optional..
@bijection function mixture_weight_1d_inv(model_args, proposal_args, proposal_retval)
    n, k = model_args
    i = @read_discrete_from_proposal(:i)
    j = @read_discrete_from_proposal(:j)
    @copy_proposal_to_proposal(:i, :i)
    @copy_proposal_to_proposal(:j, :j)
    if i == j
        return nothing
    end
    prev_mixture_weights = @read_continuous_from_model(:mixture_weights)
    prev_i_weight = prev_mixture_weights[i]
    prev_j_weight = prev_mixture_weights[j]
    total = prev_i_weight + prev_j_weight
    @write_continuous_to_proposal(:u, prev_i_weight / total)
    new_mixture_weights = copy(prev_mixture_weights)
    u = @read_continuous_from_proposal(:u)
    new_mixture_weights[i] = u * total
    new_mixture_weights[j] = (1 - u) * total
    @write_continuous_to_model(:mixture_weights, new_mixture_weights)
    return nothing
end

is_involution!(mixture_weight_1d_inv)

function mixture_weight_1d_move(trace)
    return mh(trace, mixture_weight_1d_proposal, (), mixture_weight_1d_inv; check=true)
end

#@gen function mixture_weight_random_walk(trace)
    #@assert all(trace[:mixture_weights] .> 0)
    #new_mixture_weights = ({:mixture_weights} ~ dirichlet(trace[:mixture_weights] * 10 .+ 1e-2))
    #if !all(new_mixture_weights .> 0)
        #@assert false
    #end
#end

###########################
# gibbs move on variances #
###########################

function variance_conditional_dists(trace)
    n, k = get_args(trace)
    cond_a = ones(2, k)
    cond_b = ones(2, k)
    for i in 1:n
        z = trace[(:z, i)]
        mu_x = trace[(:mu_x, z)]
        mu_y = trace[(:mu_y, z)]
        mu = [mu_x, mu_y]
        datum = [trace[(:x, i)], trace[(:y, i)]]
        cond_a[:,z] = cond_a[:,z] .+ 0.5
        cond_b[:,z] = cond_b[:,z] .+ 0.5 * (datum .- mu).^2
    end
    return (cond_a, cond_b)
end

@gen function variance_conditional_proposal(trace)
    n, k = get_args(trace)
    cond_a, cond_b = variance_conditional_dists(trace)
    for i in 1:k
        {(:var_x, i)} ~ inv_gamma(cond_a[1,i], cond_b[1,i])
        {(:var_y, i)} ~ inv_gamma(cond_a[2,i], cond_b[2,i])
    end
end

function variance_gibbs_move(trace)
    trace, acc = mh(trace, variance_conditional_proposal, ())
    @assert acc
    return trace
end

############################
# gibbs move on assignment #
############################

function assignment_conditional_dist(
        datum::Vector{Float64}, mixture_weights::Vector{Float64}, params::Vector{ClusterParams{Float64}})
    x, y = datum
    k = length(mixture_weights)
    @assert length(params) == k
    log_likelihoods = Vector{Float64}(undef, k)
    for (i, p) in enumerate(params)
        #l = logpdf(normal, x, p.mu_x, sqrt(max(1e-10, p.var_x)))
        #l += logpdf(normal, y, p.mu_y, sqrt(max(1e-10, p.var_y)))
        l = logpdf(normal, x, p.mu_x, sqrt(p.var_x))
        l += logpdf(normal, y, p.mu_y, sqrt(p.var_y))
        log_likelihoods[i] = l
    end
    log_probs = log.(mixture_weights) .+ log_likelihoods
    probs = exp.(log_probs .- logsumexp(log_probs))
    probs = probs .+ 1e-9
    return probs / sum(probs)
end

@gen function assignment_conditional_proposal(trace, i::Int)
    x = trace[(:x, i)]
    y = trace[(:y, i)]
    mixture_weights = trace[:mixture_weights]
    params = get_retval(trace)
    probs = assignment_conditional_dist([x, y], mixture_weights, params)
    {(:z, i)} ~ categorical(probs)
    return nothing
end

function assignment_gibbs_move(trace, i::Int)
    trace, acc = mh(trace, assignment_conditional_proposal, (i,))
    return trace
end

# random walk move

@gen function random_walk(trace, addr)
    {addr} ~ normal(trace[addr], 0.1)
end

function do_mcmc(trace, n_iters)
    @assert all(trace[:mixture_weights] .> 0)
    n, k = get_args(trace)
    for iter in 1:n_iters
        for i in 1:k
            trace, = mh(trace, select((:mu_x, i), (:mu_y, i)))
            trace, = mh(trace, random_walk, ((:mu_x, i),))
            trace, = mh(trace, random_walk, ((:mu_y, i),))

            trace, = mh(trace, select((:var_x, i), (:var_y, i)))
            trace, = mh(trace, random_walk, ((:var_x, i),))
            trace, = mh(trace, random_walk, ((:var_y, i),))

            trace, = mh(trace, select(:mixture_weights))
            #trace, = mh(trace, mixture_weight_random_walk, ())
            trace, = mixture_weight_1d_move(trace)
        end
        trace = variance_gibbs_move(trace)
        for i in 1:n
            trace = assignment_gibbs_move(trace, i)
        end
    end
    return trace
end

function split_variances(relative_weight_1, var, mu_1, mu_2, var_dof)
    @assert var > 0
    @assert 0 < var_dof < 1
    #C = dispersion_term(relative_weight_1, (1 - relative_weight_1) , mu_1, mu_2)
    denom = relative_weight_1 * var_dof + (1 - relative_weight_1) * (1 - var_dof)
    #println("var - C: $(var - C)")
    var_1 = var * var_dof / denom
    var_2 = var * (1 - var_dof) / denom
    @assert var_1 > 0
    @assert var_2 > 0
    return (var_1, var_2)
end

function merge_variances(var_1, var_2, relative_weight_1, mu_1, mu_2)
    #C = dispersion_term(relative_weight_1, (1 - relative_weight_1), mu_1, mu_2)
    var = relative_weight_1 * var_1 + (1 - relative_weight_1) * var_2#+ C
    dof = var_1 / (var_1 + var_2)
    @assert var > 0
    @assert 0 < dof < 1
    return (var, dof)
end

function new_mixture_weights(prev_mixture_weights::Vector{T}, cluster_to_split::Int, relative_weight_1::T, relative_weight_2::T) where {T<:Real}
    @assert all(prev_mixture_weights .> 0)
    prev_weight = prev_mixture_weights[cluster_to_split]
    k = length(prev_mixture_weights)
    arr = Vector{T}(undef, k+1)
    for i=1:k
        arr[i] = prev_mixture_weights[i]
    end
    arr[cluster_to_split] = relative_weight_1 * prev_weight
    arr[k+1] = relative_weight_2 * prev_weight
    @assert all(arr .> 0)
    return arr
end

function new_means(relative_weight_1, prev_mu, dx, dy)
    (mu1_x, mu1_y) = prev_mu .- (1 - relative_weight_1) * [dx, dy]
    (mu2_x, mu2_y) = prev_mu .+ relative_weight_1 * [dx, dy]
    return (mu1_x, mu1_y, mu2_x, mu2_y)
end

function new_params(prev_mu::AbstractVector{T}, relative_weight_1::T, dx::T, dy::T, var_x::T, var_y::T, var_x_dof::T, var_y_dof::T) where {T<:Real}
    (mu1_x, mu1_y, mu2_x, mu2_y) = new_means(relative_weight_1, prev_mu, dx, dy)
    # see https://stats.stackexchange.com/questions/16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians/16609#16609
    var1_x, var2_x = split_variances(relative_weight_1, var_x, mu1_x, mu2_x, var_x_dof)
    var1_y, var2_y = split_variances(relative_weight_1, var_y, mu1_y, mu2_y, var_y_dof)
    params1 = ClusterParams(mu1_x, mu1_y, var1_x, var1_y)
    params2 = ClusterParams(mu2_x, mu2_y, var2_x, var2_y)
    return (params1, params2)
end

@gen function q_split(trace)
    n, k = get_args(trace)

    # pick random cluster to split
    cluster_to_split ~ uniform_discrete(1, k)

    # sample degree of freedom for the mixture weights
    # the fraction of the weight for cluster_to_split that will be kept by cluster_to_split
    # (the rest of the weight of cluster_to_split will be placed on the new cluster)
    relative_weight_1 ~ beta(10, 10)

    # sample degrees of freedom for the means
    dx ~ normal(0, 1)
    dy ~ normal(0, 1)

    # sample degree of freedom for vars
    var_x_dof ~ beta(2, 2)
    var_y_dof ~ beta(2, 2)

    # for each data point in the cluster to be split, sample assignment conditially between the two new clusters
    prev_mu = [trace[(:mu_x, cluster_to_split)], trace[(:mu_y, cluster_to_split)]]
    var_x, var_y = (trace[(:var_x, cluster_to_split)], trace[(:var_y, cluster_to_split)])
    params1, params2  = new_params(prev_mu, relative_weight_1, dx, dy, var_x, var_y, var_x_dof, var_y_dof)
    for i in 1:n
        if trace[(:z, i)] == cluster_to_split
            (x, y) = (trace[(:x, i)], trace[(:y, i)])
            cond_prob = assignment_conditional_dist([x, y], [relative_weight_1, 1-relative_weight_1], [params1, params2])[2]
            {(:join_new_cluster, i)} ~ bernoulli(cond_prob)
        end
    end

    return nothing
end

@gen function q_merge(trace)
    n, k = get_args(trace)

    # pick random cluster to merge with last cluster (k)
    # TODO would it be better to pick a cluster whose data is near the data of cluster k?
    # or, even just deterministically pick the nearest cluster?
    cluster_to_merge ~ uniform_discrete(1, k-1)

    return nothing
end

using LinearAlgebra: norm

function dispersion_term(relative_weight_1, relative_weight_2, mu_1, mu_2)
    # NOTE..
    return 0.
    #return (
        #relative_weight_1 * mu_1^2 +
        #relative_weight_2 * mu_2^2 -
        #(relative_weight_1 * mu_1 + relative_weight_2 * mu_2)^2)
end

@bijection function h_split(model_args, proposal_args, proposal_retval)
    n, k = model_args
    println("k: $k")

    cluster_to_split = @read_discrete_from_proposal(:cluster_to_split)
    @write_discrete_to_proposal(:cluster_to_merge, cluster_to_split)

    # compute new mixture weights
    relative_weight_1 = @read_continuous_from_proposal(:relative_weight_1) # 1
    relative_weight_2 = 1 - relative_weight_1
    prev_mixture_weights = @read_continuous_from_model(:mixture_weights) # k
    mw = new_mixture_weights(prev_mixture_weights, cluster_to_split, relative_weight_1, relative_weight_2)
    @write_continuous_to_model(:mixture_weights, mw) # k + 1

    # compute new cluster parameters
    prev_mu_x = @read_continuous_from_model((:mu_x, cluster_to_split))
    prev_mu_y = @read_continuous_from_model((:mu_y, cluster_to_split))
    prev_mu = [prev_mu_x, prev_mu_y]
    dx = @read_continuous_from_proposal(:dx)
    dy = @read_continuous_from_proposal(:dy)
    var_x = @read_continuous_from_model((:var_x, cluster_to_split))
    var_y = @read_continuous_from_model((:var_y, cluster_to_split))
    var_x_dof = @read_continuous_from_proposal(:var_x_dof)
    var_y_dof = @read_continuous_from_proposal(:var_y_dof)
    params1, params2 = new_params(prev_mu, relative_weight_1, dx, dy, var_x, var_y, var_x_dof, var_y_dof)
    @write_continuous_to_model((:mu_x, cluster_to_split), params1.mu_x)
    @write_continuous_to_model((:mu_y, cluster_to_split), params1.mu_y)
    @write_continuous_to_model((:var_x, cluster_to_split), params1.var_x)
    @write_continuous_to_model((:var_y, cluster_to_split), params1.var_y)
    @write_continuous_to_model((:mu_x, k+1), params2.mu_x)
    @write_continuous_to_model((:mu_y, k+1), params2.mu_y)
    @write_continuous_to_model((:var_x, k+1), params2.var_x)
    @write_continuous_to_model((:var_y, k+1), params2.var_y)

    # move data points into clusters using Gibbs sampling
    for i in 1:n
        cluster = @read_discrete_from_model((:z, i))
        if cluster != cluster_to_split
            continue
        end
        if @read_discrete_from_proposal((:join_new_cluster, i))
            @write_discrete_to_model((:z, i), k+1)
        else
            @write_discrete_to_model((:z, i), cluster)
        end
    end
end

@bijection function h_merge(model_args, proposal_args, proposal_retval)
    n, k = model_args

    # the cluster to merge with cluster k
    cluster_to_merge = @read_discrete_from_proposal(:cluster_to_merge)
    @assert cluster_to_merge < k
    @write_discrete_to_proposal(:cluster_to_split, cluster_to_merge)

    # mixture weights
    prev_mixture_weights = @read_continuous_from_model(:mixture_weights) # k
    weight_1 = prev_mixture_weights[cluster_to_merge]
    weight_2 = prev_mixture_weights[k]
    relative_weight_1 = weight_1 / (weight_1 + weight_2)
    relative_weight_2 = 1 - relative_weight_1
    @write_continuous_to_proposal(:relative_weight_1, relative_weight_1)
    new_mixture_weights = prev_mixture_weights[1:k-1] # copies
    new_mixture_weights[cluster_to_merge] = weight_1 + weight_2
    @write_continuous_to_model(:mixture_weights, new_mixture_weights)

    # merged cluster mean and dofs
    mu_x_1 = @read_continuous_from_model((:mu_x, cluster_to_merge))
    mu_y_1 = @read_continuous_from_model((:mu_y, cluster_to_merge))
    mu_1 = [mu_x_1, mu_y_1]
    mu_x_2 = @read_continuous_from_model((:mu_x, k))
    mu_y_2 = @read_continuous_from_model((:mu_y, k))
    mu_2 = [mu_x_2, mu_y_2]
    prev_mu = relative_weight_1 * mu_1 .+ relative_weight_2 * mu_2
    (dx, dy) = mu_2 .- mu_1
    @write_continuous_to_model((:mu_x, cluster_to_merge), prev_mu[1])
    @write_continuous_to_model((:mu_y, cluster_to_merge), prev_mu[2])
    @write_continuous_to_proposal(:dx, dx)
    @write_continuous_to_proposal(:dy, dy)

    # compute merged variances and variance dofs
    var_x_1 = @read_continuous_from_model((:var_x, cluster_to_merge))
    var_y_1 = @read_continuous_from_model((:var_y, cluster_to_merge))
    var_x_2 = @read_continuous_from_model((:var_x, k))
    var_y_2 = @read_continuous_from_model((:var_y, k))
    var_x, var_x_dof = merge_variances(var_x_1, var_x_2, relative_weight_1, mu_x_1, mu_x_2)
    var_y, var_y_dof = merge_variances(var_y_1, var_y_2, relative_weight_1, mu_y_1, mu_y_2)
    @write_continuous_to_proposal(:var_x_dof, var_x_dof)
    @write_continuous_to_proposal(:var_y_dof, var_y_dof)
    @write_continuous_to_model((:var_x, cluster_to_merge), var_x)
    @write_continuous_to_model((:var_y, cluster_to_merge), var_y)

    # move the data points deterministically into the merged cluster
    for i in 1:n
        cluster = @read_discrete_from_model((:z, i))
        if (cluster == cluster_to_merge) || (cluster == k)
            @write_discrete_to_model((:z, i), cluster_to_merge)
            @write_discrete_to_proposal((:join_new_cluster, i), cluster == k)
        end
    end
end

pair_bijections!(h_split, h_merge)

function split_smc_step(trace, data, obs; check=false)
    n, k = get_args(trace)
    q_split_trace = simulate(q_split, (trace,))

    (new_trace, q_merge_trace, model_weight) = h_split(
        trace, q_split_trace, 
        q_merge, (),
        (n, k+1), obs; check=check, prev_observations=obs)
    weight = model_weight + get_score(q_merge_trace) - get_score(q_split_trace)
    return (new_trace, weight, q_split_trace)
end

function run_smc(data, k_max::Int, num_mcmc::Int)
    n = length(data)
    obs = choicemap()
    for i in 1:n
        obs[(:x, i)], obs[(:y, i)] = data[i]
    end

    # start with a single cluster
    traces = []
    q_split_traces = []
    trace, weight = generate(model, (n, 1), obs)

    for k=1:k_max-1
        push!(traces, trace) # record trace before mcmc

        # do some inference for current number of clusters
        trace = do_mcmc(trace, num_mcmc)
        @assert get_args(trace)[2] == k

        # record trace after mcmc
        push!(traces, trace)

        # increment the number of clusters with a split
        trace, incr_weight, q_split_trace = split_smc_step(trace, data, obs; check=true)
        println(trace[:mixture_weights])
        @assert all(trace[:mixture_weights] .> 0)
        weight += incr_weight
        @assert get_args(trace)[2] == k+1
        push!(q_split_traces, q_split_trace)
    end

    # record trace before mcmc
    push!(traces, trace)

    # do some inference for current number of clusters
    trace = do_mcmc(trace, num_mcmc)

    # record trace after mcmc
    push!(traces, trace)

    return (trace, weight, traces, q_split_traces)
end

# generate data with four clusters and 50 data points
function do_experiment()
    seed!(3)
    n = 50
    var = 0.1^2
    ground_truth_trace, = generate(model, (n, 4), choicemap(
        (:mixture_weights, 0.25 * ones(4)),
        ((:mu_x, 1), 1), ((:mu_y, 1), 1), ((:var_x, 1), var), ((:var_y, 1), var),
        ((:mu_x, 2), -1), ((:mu_y, 2), 1), ((:var_x, 2), var), ((:var_y, 2), var),
        ((:mu_x, 3), -1), ((:mu_y, 3), -1), ((:var_x, 3), var), ((:var_y, 3), var),
        ((:mu_x, 4), 1), ((:mu_y, 4), -1), ((:var_x, 4), var), ((:var_y, 4), var)))
    
    function set_xlim_ylim()
        gca().set_xlim((-2, 2))
        gca().set_ylim((-2, 2))
    end

    all_colors = ["red", "green", "blue", "orange"]
    point_size = 10
    point_alpha = 0.5
    cluster_center_size = 200

    function colors(trace)
        n, k = get_args(trace)
        @assert k <= length(all_colors)
        zs = [trace[(:z, i)] for i=1:n]
        all_colors[zs]
    end

    function show_params(trace)
        n, k = get_args(trace)
        for i=1:k
            mu_x = trace[(:mu_x, i)]
            mu_y = trace[(:mu_y, i)]
            var_x = trace[(:var_x, i)]
            var_y = trace[(:var_y, i)]
            scatter([mu_x], [mu_y], c=all_colors[i], s=cluster_center_size, marker="x")
            ellipse = matplotlib.patches.Ellipse((mu_x, mu_y), sqrt(var_x), sqrt(var_y), fill=false, edgecolor=all_colors[i])
            gca().add_artist(ellipse)
        end
    end
    
    data = Vector{Tuple{Float64,Float64}}(undef, n)
    for i in 1:n
        data[i] = (ground_truth_trace[(:x, i)], ground_truth_trace[(:y, i)])
    end
    
    k_max = 4
    num_mcmc = 200
    trace, weight, traces, q_split_traces = run_smc(data, k_max, num_mcmc)
    #display(get_choices(trace))

    figure(figsize=(9,9))
    nrows, ncols = 4, 4
    xs = [d[1] for d in data]
    ys = [d[2] for d in data]

    # ground truth
    subplot(nrows, ncols, 1)
    title("ground truth")
    scatter(xs, ys, c=colors(ground_truth_trace), s=point_size, alpha=point_alpha)
    show_params(ground_truth_trace)
    set_xlim_ylim()

    cur = 1

    # iterations..
    for k=1:k_max

        # before mcmc
        subplot(nrows, ncols, 1 + cur)
        title("k=$k initial")
        trace = traces[cur]; cur += 1
        scatter(xs, ys, c=colors(trace), s=point_size, alpha=point_alpha)
        if k > 1
            qtr = q_split_traces[k-1]
            cluster_to_split = qtr[:cluster_to_split]
            mu_x_1, mu_y_1 = (trace[(:mu_x, cluster_to_split)], trace[(:mu_y, cluster_to_split)])
            mu_x_2, mu_y_2 = (trace[(:mu_x, k)], trace[(:mu_y, k)])
            mu_x = (mu_x_1 + mu_x_2) / 2
            mu_y = (mu_y_1 + mu_y_2) / 2
            plot([mu_x_1, mu_x_2], [mu_y_1, mu_y_2], color="black", linestyle="--")
            scatter([mu_x], [mu_y], color=all_colors[cluster_to_split], marker="x", s=cluster_center_size)
        end
        show_params(trace)
        set_xlim_ylim()

        # after mcmc
        subplot(nrows, ncols, 1 + cur)
        title("k=$k after MCMC")
        trace = traces[cur]; cur += 1
        scatter(xs, ys, c=colors(trace), s=point_size, alpha=point_alpha)
        show_params(trace)
        set_xlim_ylim()

    end

    @assert length(traces) == cur - 1

    tight_layout()
    savefig("clustering.pdf")

end

do_experiment()
