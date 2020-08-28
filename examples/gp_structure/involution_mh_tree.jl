using Gen

include("../gp_structure/shared.jl")

@gen function covariance_prior()
    node_type = @trace(categorical(node_dist), :type)

    if node_type == CONSTANT
        param = @trace(uniform_continuous(0, 1), :param)
        node = Constant(param)

    # linear kernel
    elseif node_type == LINEAR
        param = @trace(uniform_continuous(0, 1), :param)
        node = Linear(param)

    # squared exponential kernel
    elseif node_type == SQUARED_EXP
        length_scale= @trace(uniform_continuous(0, 1), :length_scale)
        node = SquaredExponential(length_scale)

    # periodic kernel
    elseif node_type == PERIODIC
        scale = @trace(uniform_continuous(0, 1), :scale)
        period = @trace(uniform_continuous(0, 1), :period)
        node = Periodic(scale, period)

    # plus combinator
    elseif node_type == PLUS
        left = @trace(covariance_prior(), :left)
        right = @trace(covariance_prior(), :right)
        node = Plus(left, right)

    # times combinator
    elseif node_type == TIMES
        left = @trace(covariance_prior(), :left)
        right = @trace(covariance_prior(), :right)
        node = Times(left, right)

    # unknown node type
    else
        error("Unknown node type: $node_type")
    end

    return node
end

@gen function model(xs::Vector{Float64})
    n = length(xs)

    # sample covariance function
    covariance_fn::Node = @trace(covariance_prior(), :tree)

    # sample diagonal noise
    noise = @trace(gamma(1, 1), :noise) + 0.01

    # compute covariance matrix
    cov_matrix = compute_cov_matrix_vectorized(covariance_fn, noise, xs)

    # sample from multivariate normal
    @trace(mvnormal(zeros(n), cov_matrix), :ys)

    return covariance_fn
end

##############
# noise move #
##############

@gen function noise_proposal(prev_trace)
    @trace(gamma(1, 1), :noise)
end

noise_move(trace) = metropolis_hastings(trace, noise_proposal, ())[1]

########################
# replace subtree move #
########################

@gen function pick_random_node_path(node::Node, path::Vector{Symbol})
    if isa(node, LeafNode)
        @trace(bernoulli(1), :done)
        path
    elseif @trace(bernoulli(0.5), :done)
        path
    elseif @trace(bernoulli(0.5), :recurse_left)
        push!(path, :left)
        @trace(pick_random_node_path(node.left, path), :left)
    else
        push!(path, :right)
        @trace(pick_random_node_path(node.right, path), :right)
    end
end
@gen function subtree_proposal(prev_trace)
    prev_subtree_node::Node = get_retval(prev_trace)
    (path::Vector{Symbol}) = @trace(pick_random_node_path(prev_subtree_node, Symbol[]), :choose_subtree_root)
    new_subtree_node::Node = @trace(covariance_prior(), :subtree) # mixed discrete / continuous
    (path, new_subtree_node)
end

@involution function subtree_involution(model_args::Tuple, proposal_args::Tuple, fwd_ret::Tuple)

    (path::Vector{Symbol}, new_subtree_node) = fwd_ret

    # populate backward assignment with choice of root
    @copy_proposal_to_proposal(:choose_subtree_root, :choose_subtree_root)

    # swap subtrees
    model_subtree_addr = isempty(path) ? :tree : (:tree => foldr(=>, path))
    @copy_proposal_to_model(:subtree, model_subtree_addr)
    @copy_model_to_proposal(model_subtree_addr, :subtree)
end

replace_subtree_move(trace) = metropolis_hastings(
    trace, subtree_proposal, (), subtree_involution; check=true)[1]


#####################
# inference program #
#####################

function inference(xs::Vector{Float64}, ys::Vector{Float64}, num_iters::Int, callback)

    # observed data
    constraints = choicemap()
    constraints[:ys] = ys

    # generate initial trace consistent with observed data
    (trace, _) = generate(model, (xs,), constraints)

    # do MCMC
    local covariance_fn::Node
    local noise::Float64
    for iter=1:num_iters

        covariance_fn = get_retval(trace)
        noise = trace[:noise]
        #callback(covariance_fn, noise)

        # do MH move on the subtree
        trace = replace_subtree_move(trace)

        # do MH move on the top-level white noise
        trace = noise_move(trace)
    end
    (covariance_fn, noise)
end

function experiment()

    # load and rescale the airline dataset
    (xs, ys) = get_airline_dataset()
    xs_train = xs[1:100]
    ys_train = ys[1:100]
    xs_test = xs[101:end]
    ys_test = ys[101:end]

    # set seed
    Random.seed!(0)

    # print MSE and predictive log likelihood on test data
    callback = (covariance_fn, noise) -> begin
        pred_ll = predictive_ll(covariance_fn, noise, xs_train, ys_train, xs_test, ys_test)
        mse = compute_mse(covariance_fn, noise, xs_train, ys_train, xs_test, ys_test)
        println("mse: $mse, predictive log likelihood: $pred_ll")
    end

    # do inference, time it
    @time (covariance_fn, noise) = inference(xs_train, ys_train, 1000, callback)

    # sample predictions
    pred_ys = predict_ys(covariance_fn, noise, xs_train, ys_train, xs_test)
end

@time experiment()
@time experiment()
