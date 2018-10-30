include("shared.jl")

# do we need a separate proposal function that uses the same address space?
# it should not compute the covariance matrix itself..

# production kernel
# input type Nothing (U=Nothing)
# argdiff type: Union{NoArgDiff,Nothing} (DU=Nothing)
# return type: Tuple{Int,Vector{Nothing}} (V=Tuple{Int,Vector{Float64}},U=Nothing)
# retdiff type: TreeProductionRetDiff{NodeTypeDiff,Nothing}

# aggregation kernel
# input type: Tuple{Int,Vector{Node}} (V=Tuple{Int,Vector{Float64}},W=Node)
# argdiff type:TreeAggregationArgDiff{NodeTypeDiff,CovNodeDiff} (DV=NodeTypeDiff,DW=CovNodeDiff)
# return type: Node (W=Node)
# retdiff type: CovNodeDiff

# U = Nothing; DU = Nothing
# V = Int; DV = NodeTypeXSDiff (always change..)
# W = Node; DW = CovNodeDiff

"""
Indicates whether the node type may have changed or not.
"""
struct NodeTypeXsDiff
    same::Bool
end
Gen.isnodiff(diff::NodeTypeXsDiff) = diff.same

"""
Indicates whether the covariance function may have changed or not.
"""
struct CovMatrixDiff
    same::Bool
end
Gen.isnodiff(diff::CovMatrixDiff) = diff.same

const production_retdiff = TreeProductionRetDiff{NodeTypeXsDiff,Nothing}(
    NodeTypeXsDiff(false), Dict{Int,Nothing}())

@gen function cov_mat_production_kernel(xs::Vector{Float64})
    node_type = @addr(categorical(node_dist), :type)
    num_children = node_type_to_num_children[node_type]

    @diff @retdiff(production_retdiff)

    return ((node_type, xs), [xs for _=1:num_children])
end

struct NodeAndCovMatrix
    node::Node
    cov_matrix::Matrix{Float64}
end

@gen function cov_mat_aggregation_kernel(node_type_and_xs::Tuple{Int,Vector{Float64}},
                                         children::Vector{NodeAndCovMatrix})
    (node_type, xs) = node_type_and_xs
    local node::Node
    local cov_matrix::Matrix{Float64}

    # constant kernel
    if node_type == CONSTANT
        @assert length(children) == 0
        param = @addr(uniform_continuous(0, 1), :param)
        node = Constant(param)
        cov_matrix = eval_cov_mat(node, xs)

    # linear kernel
    elseif node_type == LINEAR
        @assert length(children) == 0
        param = @addr(uniform_continuous(0, 1), :param)
        node = Linear(param)
        cov_matrix = eval_cov_mat(node, xs)

    # squared exponential kernel
    elseif node_type == SQUARED_EXP
        @assert length(children) == 0
        length_scale = 0.01 + @addr(uniform_continuous(0, 1), :length_scale)
        node = SquaredExponential(length_scale)
        cov_matrix = eval_cov_mat(node, xs)

    # periodic kernel
    elseif node_type == PERIODIC
        @assert length(children) == 0
        scale = 0.01 + @addr(uniform_continuous(0, 1), :scale)
        period = 0.01 + @addr(uniform_continuous(0, 1), :period)
        node = Periodic(scale, period)
        cov_matrix = eval_cov_mat(node, xs)

    # plus combinator
    elseif node_type == PLUS
        @assert length(children) == 2
        node = Plus(children[1].node, children[2].node)
        cov_matrix = children[1].cov_matrix .+ children[2].cov_matrix

    # times combinator
    elseif node_type == TIMES
        @assert length(children) == 2
        node = Times(children[1].node, children[2].node)
        cov_matrix = children[1].cov_matrix .* children[2].cov_matrix

    # unknown
    else
        error("unknown node type $node_type")
    end

    @diff @retdiff(CovMatrixDiff(false))
    
    return NodeAndCovMatrix(node, cov_matrix)
end

const cov_mat_generator = Tree(cov_mat_production_kernel,
                               cov_mat_aggregation_kernel,
                               max_branch,
                               Vector{Float64}, Tuple{Int,Vector{Float64}}, NodeAndCovMatrix,
                               NodeTypeXsDiff, Nothing, CovMatrixDiff)

@gen function model(xs::Vector{Float64})

    # sample covariance matrix
    node_and_cov_matrix::NodeAndCovMatrix = @addr(cov_mat_generator(xs, 1), :tree, noargdiff)

    # sample diagonal noise
    noise = @addr(gamma(1, 1), :noise) + 0.01

    # compute covariance matrix
    n = length(xs)
    cov_matrix = node_and_cov_matrix.cov_matrix + noise * eye(n)

    # sample from multivariate normal   
    @addr(mvnormal(zeros(n), cov_matrix), :ys)

    return (node_and_cov_matrix.node, cov_matrix)
end

@gen function covariance_proposal_recurse(cur::Int)
    
    node_type = @addr(categorical(node_dist), (cur, Val(:production)) => :type)

    base_addr = (cur, Val(:aggregation))
    if node_type == CONSTANT
        @addr(uniform_continuous(0, 1), base_addr => :param)

    # linear kernel
    elseif node_type == LINEAR
        @addr(uniform_continuous(0, 1), base_addr => :param)

    # squared exponential kernel
    elseif node_type == SQUARED_EXP
        @addr(uniform_continuous(0, 1), base_addr => :length_scale)

    # periodic kernel
    elseif node_type == PERIODIC
        @addr(uniform_continuous(0, 1), base_addr => :scale)
        @addr(uniform_continuous(0, 1), base_addr => :period)

    # plus combinator
    elseif node_type == PLUS
        child1 = get_child(cur, 1, max_branch)
        child2 = get_child(cur, 2, max_branch)
        @splice(covariance_proposal_recurse(child1))
        @splice(covariance_proposal_recurse(child2))

    # times combinator
    elseif node_type == TIMES
        child1 = get_child(cur, 1, max_branch)
        child2 = get_child(cur, 2, max_branch)
        @splice(covariance_proposal_recurse(child1))
        @splice(covariance_proposal_recurse(child2))

    # unknown node type
    else
        error("Unknown node type: $node_type")
    end
end

@gen function subtree_proposal(prev_trace, root::Int)
    @addr(covariance_proposal_recurse(root), :tree, noargdiff)
end

@gen function noise_proposal(prev_trace)
    @addr(gamma(1, 1), :noise)
end

function inference(xs, ys, num_iters::Int)
    constraints = DynamicAssignment()
    constraints[:ys] = ys
    (trace, _) = generate(model, (xs,), constraints)
    local covariance_fn::Node
    local cov_matrix::Matrix{Float64}
    local noise::Float64
    for iter=1:num_iters
        # pick a node to expand
        (covariance_fn, cov_matrix) = get_call_record(trace).retval
        root = pick_random_node(covariance_fn, 1, max_branch)
        @assert has_internal_node(get_assignment(trace), :tree => (root, Val(:production)))
        trace = mh(model, subtree_proposal, (root,), trace)
        trace = mh(model, noise_proposal, (), trace) # in principle, doesn't require recomputing the covariance matrix (but in our version it does..) TODO change that?
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
    savefig("incremental.png")
end

experiment()
