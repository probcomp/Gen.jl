using Gen
using PyPlot
using Random: seed!
using ForwardDiff

#########
# model #
#########

@gen function model(n:Int, k::Int)
    # n is number of data points
    # k is the number of clusters

    # prior on the mixture weights
    mixture_weights ~ dirichlet(ones(k))

    # sample cluster parameters
    params = Vector{Tuple{Float64,Float64,Float64,Float64}}(undef, k)
    for i=1:k
        mu_x = ({(:mu_x, i)} ~ normal(0, 10))
        mu_y = ({(:mu_y, i)} ~ normal(0, 10))
        var_x = ({(:var_x, i)} ~ inv_gamma(1, 1)) # was 0.1
        var_y = ({(:var_y, i)} ~ inv_gamma(1, 1)) # was 0.1
        params[i] = (mu_x, mu_y, var_x, var_y)
    end

    # sample assignments and data
    for i=1:n
        z = ({(:z, i)} ~ categorical(mixture_weights))
        mu_x, mu_y, var_x, var_y = params[z]
        {(:x, i)} ~ normal(mu_x, sqrt(var_x))
        {(:y, i)} ~ normal(mu_y, sqrt(var_y))
    end
    return nothing
end

################################
# MCMC move on mixture weights #
################################

#@gen function mixture_weight_prop(trace)
    #n, k = get_args(trace)
    ## pick two clusters whose weight should change
    #i ~ uniform_discrete(1, k)
    #j ~ uniform_discrete(1, k)
    #if i != j
        #u ~ uniform_continous(0, 1)
    #end
    #return nothing
#end
#
## TODO make these tree arguments optional..
#@bijection function mixture_weight_inv(model_args, proposal_args, proposal_retval)
    #i = @read_discrete_from_proposal(:i)
    #j = @read_discrete_from_proposal(:j)
    #@copy_proposal_to_proposal(:i, :i)
    #@copy_proposal_to_proposal(:j, :j)
    #if i == j
        #return nothing
    #end
    ## TODO
    #return nothing
#end

@gen function mixture_weight_random_walk(trace)
    {:mixture_weights} ~ dirichlet(trace[:mixture_weights] * 100)
end

# random walk move

@gen function random_walk(trace, addr)
    {addr} ~ normal(trace[addr], 0.1)
end

function do_mcmc(trace, n_iters)
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
            trace, = mh(trace, mixture_weight_random_walk, ())
        end
        for i in 1:n
            # TODO replace these with gibbs moves
            trace, = mh(trace, select((:z, i)))
        end
    end
    trace
end

@gen function q_split(trace, data::Vector{Tuple{Float64,Float64}})
    n, k = get_args(trace)

    # pick random cluster to split
    cluster_to_split ~ uniform_discrete(1, k)

    # sample degree of freedom for the mixture weights
    mixture_weight_dof ~ uniform_continuous(0, 1)

    # sample degrees of freedom for the means
    dx ~ normal(0, 1)
    dy ~ normal(0, 1)

    # sample degree of freedom for vars
    var_x_dof ~ uniform_continuous(0, 1)
    var_y_dof ~ uniform_continuous(0, 1)

    # new var values
    #var1 ~ inv_gamma(1, 1)
    #var2 ~ inv_gamma(1, 1)

    # for each data point in the cluster to be split, sample a biased bernoulli
    # TODO actually sample from the conditional distribution
    for i in 1:n
        if trace[(:z, i)] == cluster_to_split
            {(:cluster_choice, i)} ~ bernoulli(0.9)
        end
    end
end

@gen function q_merge(trace, data::Vector{Tuple{Float64,Float64}})
    n, k = get_args(trace)

    # pick random cluster to merge with last cluster (k)
    # TODO would it be better to pick a cluster whose data is near the data of cluster k?
    # or, even just deterministically pick the nearest cluster?
    cluster_to_merge ~ uniform_discrete(1, k-1)

    merged_var ~ inv_gamma(1, 1)
end

using LinearAlgebra: norm

function dispersion_term(relative_weight_1, relative_weight_2, var, new_mu_1, new_mu_2)
    return (
        relative_weight_1 * new_mu_1^2 +
        relative_weight_2 * new_mu_2^2 -
        (relative_weight_1 * new_mu_1 + relative_weight_2 * new_mu_2)^2)
end

function split_variances(relative_weight_1, relative_weight_2, var, new_mu_1, new_mu_2, var_dof)
    C = dispersion_term(relative_weight_1, relative_weight_2, var, new_mu_1, new_mu_2)
    denom = relative_weight_1 * var_dof + relative_weight_2 * (1 - var_dof)
    var_1 = (var - C) * var_dof / denom
    var_2 = (var - C) * (1 - var_dof) / denom
    return (var_1, var_2)
end

function merge_variances(var_1, var_2, relative_weight_1, relative_weight_2, mu_1, mu_2)
    C = dispersion_term(relative_weight_1, relative_weight_2, var, new_mu_1, new_mu_2)
    var = relative_weight_1 * var_1 + relative_weight_2 * var_2 + C
    dof = var_1 / (var_1 + var_2)
    return (var, dof)
end

@bijection function h_split(model_args, proposal_args, proposal_retval)
    n, k = model_args
    data, = proposal_args

    cluster_to_split = @read_discrete_from_proposal(:cluster_to_split)
    @write_discrete_to_proposal(:cluster_to_merge, cluster_to_split)

    # compute new mixture weights
    relative_weight_1 = @read_continuous_from_proposal(:mixture_weight_dof) # 1
    relative_weight_2 = 1 - relative_weight_1
    prev_mixture_weights = @read_continuous_from_model(:mixture_weights) # k
    prev_weight = prev_mixture_weights[cluster_to_split] 
    new_mixture_weights = copy(prev_mixture_weights)
    new_mixture_weights[cluster_to_split] = relative_weight_1 * prev_weight
    push!(new_mixture_weights, relative_weight_2 * prev_weight)
    @assert length(new_mixture_weights) == k+1
    @write_continuous_to_model(:mixture_weights, new_mixture_weights) # k + 1

    # compute new means
    prev_mu_x = @read_continuous_from_model((:mu_x, cluster_to_split))
    prev_mu_y = @read_continuous_from_model((:mu_y, cluster_to_split))
    prev_mu = [prev_mu_x, prev_mu_y]
    dx = @read_continuous_from_proposal(:dx)
    dy = @read_continuous_from_proposal(:dy)
    #(new_mu_x_1, new_mu_y_1) = (prev_mu_x - dx/2, prev_mu_y - dy/2)
    #(new_mu_x_2, new_mu_y_2) = (prev_mu_x + dx/2, prev_mu_y + dy/2)
    new_mu_1 = prev_mu .- relative_weight_2 * [dx, dy]
    new_mu_2 = prev_mu .+ relative_weight_1 * [dx, dy]
    @write_continuous_to_model((:mu_x, cluster_to_split), new_mu_1[1])
    @write_continuous_to_model((:mu_y, cluster_to_split), new_mu_1[2])
    @write_continuous_to_model((:mu_x, k+1), new_mu_2[1])
    @write_continuous_to_model((:mu_y, k+1), new_mu_2[2])

    # compute new vars
    # see https://stats.stackexchange.com/questions/16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians/16609#16609
    var_x_dof = @read_continuous_from_proposal(:var_x_dof)
    var_y_dof = @read_continuous_from_proposal(:var_y_dof)
    var_x = @read_continuous_from_model((:var_x, cluster_to_split))
    var_y = @read_continuous_from_model((:var_y, cluster_to_split))
    var_x_1, var_x_2 = split_variances(relative_weight_1, relative_weight_2, var_x, new_mu_1[1], new_mu_2[1], var_x_dof)
    var_y_1, var_y_2 = split_variances(relative_weight_1, relative_weight_2, var_y, new_mu_1[2], new_mu_2[2], var_y_dof)
    @write_continuous_to_model((:var_x, cluster_to_split), var_x_1)
    @write_continuous_to_model((:var_y, cluster_to_split), var_y_1)
    @write_continuous_to_model((:var_x, k+1), var_x_2)
    @write_continuous_to_model((:var_y, k+1), var_y_2)
    #@copy_proposal_to_model(:var1, (:var, cluster_to_split))
    #@copy_proposal_to_model(:var2, (:var, k+1))
    #@copy_model_to_proposal((:var, cluster_to_split), :merged_var)

    # move data points into clusters using Gibbs sampling
    for i in 1:n
        cluster = @read_discrete_from_model((:z, i))
        if cluster != cluster_to_split
            #@write_discrete_to_model((:z, i), cluster)
            continue
        end
        (x, y) = data[i]
        dist_1 = norm([x, y] - [new_mu_x_1, new_mu_y_1])
        dist_2 = norm([x, y] - [new_mu_x_2, new_mu_y_2])
        choice = @read_discrete_from_proposal((:cluster_choice, i)) # biased to be true
        closer_to_1 = (dist_1 < dist_2)
        if (closer_to_1 && choice) || (!closer_to_1 && !choice)
            # stay in this cluster
            @write_discrete_to_model((:z, i), cluster)
        else
            # join the new cluster
            @write_discrete_to_model((:z, i), k+1)
        end
    end
end

@bijection function h_merge(model_args, proposal_args, proposal_retval)
    n, k = model_args
    data, = proposal_args

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
    @write_continuous_to_proposal(:mixture_weight_dof, relative_weight_1)
    new_mixture_weights = prev_mixture_weights[1:k-1] # copies
    new_mixture_weights[cluster_to_merge] = weight_1 + weight_2
    @write_continuous_to_model(:mixture_weights, new_mixture_weights)

    # merged cluster mean and dofs
    mu_x_1 = @read_continuous_from_model((:mu_x, cluster_to_merge))
    mu_y_1 = @read_continuous_from_model((:mu_y, cluster_to_merge))
    mu_x_2 = @read_continuous_from_model((:mu_x, k))
    mu_y_2 = @read_continuous_from_model((:mu_y, k))
    prev_mu = relative_weight_1 * [mu_x_1, mu_y_1] .+ relative_weight_2 * [mu_x_2, mu_y_2]
    (dx, dy) = [mu_x_2, mu_y_2] .- [mu_x_1, mu_y_1]
    @write_continuous_to_model((:mu_x, cluster_to_merge), prev_mu[1])
    @write_continuous_to_model((:mu_y, cluster_to_merge), prev_mu[2])
    @write_continuous_to_proposal(:dx, dx)
    @write_continuous_to_proposal(:dy, dy)

    # compute merged variances and variance dofs
    var_x_1 = @read_continuous_from_model((:var_x, cluster_to_merge))
    var_y_1 = @read_continuous_from_model((:var_y, cluster_to_merge))
    var_x_2 = @read_continuous_from_model((:var_x, k))
    var_y_2 = @read_continuous_from_model((:var_y, k))
    var_x, var_x_dof = merge_variances(var_x_1, var_x_2, relative_weight_1, relative_weight_2, mu_x_1, mu_x_2)
    var_y, var_y_dof = merge_variances(var_y_1, var_y_2, relative_weight_1, relative_weight_2, mu_y_1, mu_y_2)
    @write_continuous_to_proposal(:var_x_dof, var_x_dof)
    @write_continuous_to_proposal(:var_y_dof, var_y_dof)
    @write_continuous_to_model((:var_x, cluster_to_merge), var_x)
    @write_continuous_to_model((:var_y, cluster_to_merge), var_y)
    #@copy_model_to_proposal((:var, cluster_to_merge), :var1)
    #@copy_model_to_proposal((:var, k), :var2)
    #@copy_proposal_to_model(:merged_var, (:var, cluster_to_merge))

    # move the data points deterministically into the merged cluster
    for i in 1:n
        cluster = @read_discrete_from_model((:z, i))
        if (cluster == cluster_to_merge) || (cluster == k)
            @write_discrete_to_model((:z, i), cluster_to_merge)
            (x, y) = data[i]
            dist_1 = norm([x, y] - [mu_x_1, mu_y_1])
            dist_2 = norm([x, y] - [mu_x_2, mu_y_2])
            closer_to_1 = (dist_1 < dist_2)
            @write_discrete_to_proposal(
                (:cluster_choice, i),
                closer_to_1 == (cluster == cluster_to_merge))
        #else
            #@write_discrete_to_model((:z, i), cluster)
        end
    end
end

pair_bijections!(h_split, h_merge)

function split_smc_step(trace, data, obs; check=false)
    n, k = get_args(trace)
    q_split_trace = simulate(q_split, (trace, data))

    (new_trace, q_merge_trace, model_weight) = h_split(
        trace, q_split_trace, 
        q_merge, (data,),
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
    seed!(1)
    n = 50
    ground_truth_trace, = generate(model, (n, 4), choicemap(
        ((:mu_x, 1), 1), ((:mu_y, 1), 1), ((:var, 1), 0.1),
        ((:mu_x, 2), -1), ((:mu_y, 2), 1), ((:var, 2), 0.1),
        ((:mu_x, 3), -1), ((:mu_y, 3), -1), ((:var, 3), 0.1),
        ((:mu_x, 4), 1), ((:mu_y, 4), -1), ((:var, 4), 0.1)))
    
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
            var = trace[(:var, i)]
            scatter([mu_x], [mu_y], c=all_colors[i], s=cluster_center_size, marker="x")
            circle = matplotlib.patches.Circle((mu_x, mu_y), radius=var, fill=false, edgecolor=all_colors[i])
            gca().add_artist(circle)
        end
    end
    
    data = Vector{Tuple{Float64,Float64}}(undef, n)
    for i in 1:n
        data[i] = (ground_truth_trace[(:x, i)], ground_truth_trace[(:y, i)])
    end
    
    k_max = 4
    num_mcmc = 100
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
