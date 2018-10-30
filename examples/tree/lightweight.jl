include("shared.jl")

@gen function generate_covariance_fn(cur::Int)
    node_type = @addr(categorical(node_dist), (cur, Val(:production)) => :type)

    base_addr = (cur, Val(:aggregation))
    if node_type == CONSTANT
        param = @addr(uniform_continuous(0, 1), base_addr => :param)
        node = Constant(param)

    # linear kernel
    elseif node_type == LINEAR
        param = @addr(uniform_continuous(0, 1), base_addr => :param)
        node = Linear(param)

    # squared exponential kernel
    elseif node_type == SQUARED_EXP
        length_scale= @addr(uniform_continuous(0, 1), base_addr => :length_scale)
        node = SquaredExponential(length_scale)

    # periodic kernel
    elseif node_type == PERIODIC
        scale = @addr(uniform_continuous(0, 1), base_addr => :scale)
        period = @addr(uniform_continuous(0, 1), base_addr => :period)
        node = Periodic(scale, period)

    # plus combinator
    elseif node_type == PLUS
        child1 = get_child(cur, 1, max_branch)
        child2 = get_child(cur, 2, max_branch)
        left = @splice(generate_covariance_fn(child1))
        right = @splice(generate_covariance_fn(child2))
        node = Plus(left, right)

    # times combinator
    elseif node_type == TIMES
        child1 = get_child(cur, 1, max_branch)
        child2 = get_child(cur, 2, max_branch)
        left = @splice(generate_covariance_fn(child1))
        right = @splice(generate_covariance_fn(child2))
        node = Times(left, right)

	# unknown node type
    else
        error("Unknown node type: $node_type")
    end

	return node
end

@gen function model(xs::Vector{Float64})
	n = length(xs)

    # sample covariance matrix
	covariance_fn::Node = @addr(generate_covariance_fn(1), :tree)

    # sample diagonal noise
    noise = @addr(gamma(1, 1), :noise) + 0.01

    # compute covariance matrix
    cov_matrix = compute_cov_matrix_vectorized(covariance_fn, noise, xs)

    # sample from multivariate normal   
    @addr(mvnormal(zeros(n), cov_matrix), :ys)

    return covariance_fn
end

@gen function subtree_proposal(prev_trace, root::Int)
	@addr(generate_covariance_fn(root), :tree)
end

@gen function noise_proposal(prev_trace)
    @addr(gamma(1, 1), :noise)
end

function inference(xs, ys, num_iters::Int)
    constraints = DynamicAssignment()
    constraints[:ys] = ys
    (trace, _) = generate(model, (xs,), constraints)
    local covariance_fn::Node
    local noise::Float64
    for iter=1:num_iters

        # pick a node to expand
        covariance_fn = get_call_record(trace).retval
        root = pick_random_node(covariance_fn, 1, max_branch)

		# propose a new subtree
		# TODO correct the accept/reject ratio based on the number of nodes?
        trace = mh(model, subtree_proposal, (root,), trace)

		# propose a change to the noise level
        trace = mh(model, noise_proposal, (), trace)
    end
	
	noise = get_assignment(trace)[:noise]
    return (covariance_fn, noise)
end

function experiment()
	(xs, ys) = get_airline_dataset()
    new_xs = collect(range(0, stop=1.5, length=200))
    figure(figsize=(32,32))
    for i=1:16
        subplot(4, 4, i)
        tic()
        (covariance_fn, noise) = inference(xs, ys, 1000)
        toc()
        new_ys = predict_ys(covariance_fn, noise, xs, ys, new_xs)
        plot(xs, ys, color="black")
        plot(new_xs, new_ys, color="red")
        gca()[:set_xlim]((0, 1.5))
        gca()[:set_ylim]((-3, 3))
    end
    savefig("lightweight.png")
end

# experiment()
