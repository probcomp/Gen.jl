using Gen
using PyPlot
using Random: seed!

@gen function model(n:Int, k::Int)
    # n is number of data points
    # k is the number of clusters

    # sample cluster parameters
    params = Vector{Tuple{Float64,Float64,Float64}}(undef, k)
    for i=1:k
        mu_x = ({(:mu_x, i)} ~ normal(0, 10))
        mu_y = ({(:mu_y, i)} ~ normal(0, 10))
        #noise = ({(:noise, i)} ~ inv_gamma(1, 1))
        noise = 0.1
        params[i] = (mu_x, mu_y, noise)
    end

    # sample assignments and data
    for i=1:n
        z = ({(:z, i)} ~ uniform_discrete(1, k))
        mu_x, mu_y, noise = params[z]
        {(:x, i)} ~ normal(mu_x, noise)
        {(:y, i)} ~ normal(mu_y, noise)
    end
end

# to go from k to k+1 clusters..
# should be in two-dimensions (x, y) for plotting purposes
# should be uncollapsed (we can draw the cluster parametrs with ellipses)

# we can do mcmc inference for some number k
# (this can be hand-computed Gibbs moves on the assignments, and gradient-based parameter moves..)

@gen function random_walk(trace, addr)
    {addr} ~ normal(trace[addr], 0.1)
end

function do_mcmc(trace, n_iters)
    n, k = get_args(trace)
    for iter in 1:n_iters
        for i in 1:k
            #trace, = mh(trace, select((:noise, i)))
            trace, = mh(trace, select((:mu_x, i), (:mu_y, i)))
            trace, mh(trace, random_walk, ((:mu_x, i),))
            trace, mh(trace, random_walk, ((:mu_y, i),))
        end
        for i in 1:n
            trace, = mh(trace, select((:z, i)))
        end
    end
    trace
end

# now, we want to do split SMC...
# from k to k + 1
# we could just introduce a new cluster.

# split converts one mixture component into two, dividing the observations between the two components

# smart dumb, dumb smart: https://people.eecs.berkeley.edu/~russell/papers/uai15-sdds.pdf
# state of the art is 'restricted Gibbs split-merge (RGSM)' of Jain and Neal 2004

# should we use smart split and dumb merge?
# is this a good idea re backwards kernels and KL?
# what we want for KL is that the proposal has broad KL rel to the model
# this means, we want the backwards kernel to have small KL?

# the dumb merge move involves proposes merging random pairs of clusters.

@gen function q_split(trace, data::Vector{Tuple{Float64,Float64}})
    n, k = get_args(trace)

    # pick random cluster to split
    cluster_to_split ~ uniform_discrete(1, k)

    # sample degrees of freedom for the split to the parameters
    dx ~ normal(0, 1)
    dy ~ normal(0, 1)

    # for each data point in the cluster to be split, sample a biased bernoulli
    for i in 1:n
        if trace[(:z, i)] == cluster_to_split
            {(:cluster_choice, i)} ~ bernoulli(0.9)
        end
    end
end

@gen function q_merge(trace, data::Vector{Tuple{Float64,Float64}})
    n, k = get_args(trace)

    # pick random cluster to merge with last cluster (k)
    cluster_to_split ~ uniform_discrete(1, k-1)
end

using LinearAlgebra: norm

# TODO use the first principle component to determine the splitting direction, and choose random separation?
# but i'm not sure about the invertibility of this, ...

# TODO use update, and with change to args..
# (so we don't need to write every new value...)

# TODO let us run the backwards part of the bijection for checking purposes..

# TODO figure out what range need to be..., udpate the math to reflect this..
# (the range does not need to be everything, or does it?)
# it there an additional requirement for the MCMC case?

@bijection function h_split(model_args, proposal_args, proposal_retval)
    n, k = model_args
    data = proposal_args[2] # TODO proposal_args should not include the model trace here; call it 'extra proposal args'?

    cluster_to_split = @read_discrete_from_proposal(:cluster_to_split)
    @write_discrete_to_proposal(:cluster_to_merge, cluster_to_split)
    prev_mu_x = @read_continuous_from_model((:mu_x, cluster_to_split))
    prev_mu_y = @read_continuous_from_model((:mu_y, cluster_to_split))
    dx = @read_continuous_from_proposal(:dx)
    dy = @read_continuous_from_proposal(:dy)
    (new_mu_x_1, new_mu_y_1) = (prev_mu_x + dx/2, prev_mu_y + dy/2)
    (new_mu_x_2, new_mu_y_2) = (prev_mu_x - dx/2, prev_mu_y - dy/2)
    @write_continuous_to_model((:mu_x, cluster_to_split), new_mu_x_1)
    @write_continuous_to_model((:mu_y, cluster_to_split), new_mu_y_1)
    @write_continuous_to_model((:mu_x, k+1), new_mu_x_2)
    @write_continuous_to_model((:mu_y, k+1), new_mu_y_2)

    # copy over unchanged clusters
    for i in 1:k
        if i == cluster_to_split
            continue
        end
        @copy_model_to_model((:mu_x, i), (:mu_x, i))
        @copy_model_to_model((:mu_y, i), (:mu_y, i))
        #@copy_model_to_model((:noise, i), (:noise, i))
    end

    # move data points into clusters using Gibbs sampling
    for i in 1:n
        cluster = @read_discrete_from_model((:z, i))
        if cluster != cluster_to_split
            @write_discrete_to_model((:z, i), cluster)
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

    # the cluster to merge with cluster k
    cluster_to_merge = @read_discrete_from_proposal(:cluster_to_merge)
    @write_discrete_to_proposal(:cluster_to_split, cluster_to_merge)

    # now, we have two cluster parameters..
    mu_x_1 = @read_continuous_from_model((:mu_x, cluster_to_merge))
    mu_y_1 = @read_continuous_from_model((:mu_y, cluster_to_merge))
    mu_x_2 = @read_continuous_from_model((:mu_x, k))
    mu_y_2 = @read_continuous_from_model((:mu_y, k))

    # and we need to produce (i) one cluster parameter, and dx
    prev_mu_x = (mu_x_1 + mu_x_2) / 2
    prev_mu_y = (mu_y_1 + mu_y_2) / 2
    dx = mu_x_2 - mu_x_1
    dy = mu_y_2 - mu_y_1
    @write_continuous_to_model((:mu_x, cluster_to_merge), prev_mu_x)
    @write_continuous_to_model((:mu_y, cluster_to_merge), prev_mu_y)
    @write_continuous_to_model(:dx, dx)
    @write_continuous_to_model(:dy, dy)

    # copy over unchanged clusters
    for i in 1:k-1
        if i == cluster_to_merge
            continue
        end
        @copy_model_to_model((:mu_x, i), (:mu_x, i))
        @copy_model_to_model((:mu_y, i), (:mu_y, i))
        #@copy_model_to_model((:noise, i), (:noise, i))
    end

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
        else
            @write_discrete_to_model((:z, i), cluster)
        end
    end
end

function split_smc_step(trace, data, obs)
    n, k = get_args(trace)
    q_split_trace = simulate(q_split, (trace, data))
    (new_trace, q_merge_choices, model_weight) = h_split(
        trace, q_split_trace, model, (n, k+1), obs)
    q_merge_trace, = generate(q_merge, (new_trace, data), q_merge_choices)
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
        trace, incr_weight, q_split_trace = split_smc_step(trace, data, obs)
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
    
        # cluster centers
        ((:mu_x, 1), 1), ((:mu_y, 1), 1),
        ((:mu_x, 2), -1), ((:mu_y, 2), 1),
        ((:mu_x, 3), -1), ((:mu_y, 3), -1),
        ((:mu_x, 4), 1), ((:mu_y, 4), -1),
    
        # cluster noises)
        #((:noise, 1), 0.2),
        #((:noise, 2), 0.2),
        #((:noise, 3), 0.2),
        #((:noise, 4), 0.2)
        ))
    
    function set_xlim_ylim()
        gca().set_xlim((-2, 2))
        gca().set_ylim((-2, 2))
    end

    all_colors = ["red", "green", "blue", "orange"]

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
            scatter([mu_x], [mu_y], c=all_colors[i], s=100, marker="x")
        end
    end
    
    data = Vector{Tuple{Float64,Float64}}(undef, n)
    for i in 1:n
        data[i] = (ground_truth_trace[(:x, i)], ground_truth_trace[(:y, i)])
    end
    
    k_max = 4
    num_mcmc = 1000
    trace, weight, traces, q_split_traces = run_smc(data, k_max, num_mcmc)
    #display(get_choices(trace))

    figure(figsize=(16,16))
    nrows, ncols = 4, 4
    xs = [d[1] for d in data]
    ys = [d[2] for d in data]

    # ground truth
    subplot(nrows, ncols, 1)
    title("ground truth")
    scatter(xs, ys, c=colors(ground_truth_trace))
    show_params(ground_truth_trace)
    set_xlim_ylim()

    cur = 1

    # iterations..
    for k=1:k_max


        # before mcmc
        subplot(nrows, ncols, 1 + cur)
        title("trace before mcmc")
        trace = traces[cur]; cur += 1
        scatter(xs, ys, c=colors(trace))
        if k > 1
            qtr = q_split_traces[k-1]
            #dx, dy = qtr[:dx], qtr[:dy]
            cluster_to_split = qtr[:cluster_to_split]
            mu_x_1, mu_y_1 = (trace[(:mu_x, cluster_to_split)], trace[(:mu_y, cluster_to_split)])
            mu_x_2, mu_y_2 = (trace[(:mu_x, k)], trace[(:mu_y, k)])
            mu_x = (mu_x_1 + mu_x_2) / 2
            mu_y = (mu_y_1 + mu_y_2) / 2
            plot([mu_x_1, mu_x_2], [mu_y_1, mu_y_2], color="black", linestyle="--")
            scatter([mu_x], [mu_y], color=all_colors[k-1], marker="x", s=100)
        end
        show_params(trace)
        set_xlim_ylim()

        # after mcmc
        subplot(nrows, ncols, 1 + cur)
        title("trace after mcmc")
        trace = traces[cur]; cur += 1
        scatter(xs, ys, c=colors(trace))
        show_params(trace)
        set_xlim_ylim()

    end

    @assert length(traces) == cur - 1

    tight_layout()
    savefig("clustering.png")

end

do_experiment()
