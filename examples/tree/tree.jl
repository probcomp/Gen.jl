using Gen
using Gen: get_child
using LinearAlgebra: eye
import CSV
using PyPlot: figure, subplot, plot, scatter, gca, savefig
import Random

# TODO: figure out the correction for the probablity of picking the given node
# in the forward and backward proposal

# TODO: experiment with 'resimulation'

#########################
# load airline data set #
#########################

function get_airline_dataset()
    df = CSV.read("airline.csv")
    xs = df[1]
    ys = df[2]
    xs -= minimum(xs) # set x minimum to 0.
    xs /= maximum(xs) # scale x so that maximum is at 1.
    ys -= mean(ys) # set y mean to 0.
    ys *= 4 / (maximum(ys) - minimum(ys)) # make it fit in the window [-2, 2]
	return (xs, ys)
end

#figure(figsize=(16,16))
#subplot(4, 4, 1)
#plot(xs, ys)
#gca()[:set_xlim]((0, 1))
#gca()[:set_ylim]((-3, 3))
#savefig("airline.png")


################################
# abstract covariance function #
################################

abstract type Node end

abstract type LeafNode <: Node end

size(::LeafNode) = 1

abstract type BinaryOpNode <: Node end

size(node::BinaryOpNode) = node.size

pick_random_node(node::LeafNode, cur::Int, max_branch::Int) = cur

function pick_random_node(node::BinaryOpNode, cur::Int, max_branch::Int)
    if bernoulli(0.5)
        # pick this node
        cur
    else
        # recursively pick from the subtrees
        if bernoulli(0.5)
            pick_random_node(node.left, get_child(cur, 1, max_branch), max_branch)
        else
            pick_random_node(node.right, get_child(cur, 2, max_branch), max_branch)
        end
    end
end

"""
Constant kernel
"""
struct Constant <: LeafNode
    param::Float64
end

eval_cov(node::Constant, x1, x2) = node.param

function eval_cov_mat(node::Constant, xs::Vector{Float64})
    n = length(xs)
    fill(node.param, (n, n))
end

"""
Linear kernel
"""
struct Linear <: LeafNode
    param::Float64
end

eval_cov(node::Linear, x1, x2) = (x1 - node.param) * (x2 - node.param)

function eval_cov_mat(node::Linear, xs::Vector{Float64})
    xs_minus_param = xs .- node.param
    xs_minus_param * xs_minus_param'
end


"""
Squared exponential kernel
"""
struct SquaredExponential <: LeafNode
    length_scale::Float64
end

function eval_cov(node::SquaredExponential, x1, x2)
    exp(-0.5 * (x1 - x2) * (x1 - x2) / node.length_scale)
end

function eval_cov_mat(node::SquaredExponential, xs::Vector{Float64})
    diff = xs .- xs'
    exp.(-0.5 .* diff .* diff ./ node.length_scale)
end


"""
Periodic kernel
"""
struct Periodic <: LeafNode
    scale::Float64
    period::Float64
end

function eval_cov(node::Periodic, x1, x2)
    freq = 2 * pi / node.period
    exp((-1/node.scale) * (sin(freq * abs(x1 - x2)))^2)
end

function eval_cov_mat(node::Periodic, xs::Vector{Float64})
    freq = 2 * pi / node.period
    abs_diff = abs.(xs .- xs')
    exp.((-1/node.scale) .* (sin.(freq .* abs_diff)).^2)
end


"""
Plus node
"""
struct Plus <: BinaryOpNode
    left::Node
    right::Node
    size::Int
end

Plus(left, right) = Plus(left, right, size(left) + size(right) + 1)

function eval_cov(node::Plus, x1, x2)
    eval_cov(node.left, x1, x2) + eval_cov(node.right, x1, x2)
end

function eval_cov_mat(node::Plus, xs::Vector{Float64})
    eval_cov_mat(node.left, xs) .+ eval_cov_mat(node.right, xs)
end


"""
Times node
"""
struct Times <: BinaryOpNode
    left::Node
    right::Node
    size::Int
end

Times(left, right) = Times(left, right, size(left) + size(right) + 1)

function eval_cov(node::Times, x1, x2)
    eval_cov(node.left, x1, x2) * eval_cov(node.right, x1, x2)
end

function eval_cov_mat(node::Times, xs::Vector{Float64})
    eval_cov_mat(node.left, xs) .* eval_cov_mat(node.right, xs)
end


function compute_cov_matrix(covariance_fn::Node, noise, xs)
    n = length(xs)
    cov_matrix = Matrix{Float64}(undef, n, n)
    for i=1:n
        for j=1:n
            cov_matrix[i, j] = eval_cov(covariance_fn, xs[i], xs[j])
        end
        cov_matrix[i, i] += noise
    end
    return cov_matrix
end

function compute_cov_matrix_vectorized(covariance_fn, noise, xs)
    n = length(xs)
    eval_cov_mat(covariance_fn, xs) + noise * eye(n)
end

function compute_log_likelihood(cov_matrix::Matrix{Float64}, ys::Vector{Float64})
    n = length(ys)
    logpdf(mvnormal, ys, zeros(n), cov_matrix)
end

function predict_ys(covariance_fn::Node, noise::Float64,
                    xs::Vector{Float64}, ys::Vector{Float64},
                    new_xs::Vector{Float64})
    n_prev = length(xs)
    n_new = length(new_xs)
    means = zeros(n_prev + n_new)
    cov_matrix = compute_cov_matrix(covariance_fn, noise, vcat(xs, new_xs))
    cov_matrix_11 = cov_matrix[1:n_prev, 1:n_prev]
    cov_matrix_22 = cov_matrix[n_prev+1:n_prev+n_new, n_prev+1:n_prev+n_new]
    cov_matrix_12 = cov_matrix[1:n_prev, n_prev+1:n_prev+n_new]
    cov_matrix_21 = cov_matrix[n_prev+1:n_prev+n_new, 1:n_prev]
    @assert cov_matrix_12 == cov_matrix_21'
    mu1 = means[1:n_prev]
    mu2 = means[n_prev+1:n_prev+n_new]
    conditional_mu = mu2 + cov_matrix_21 * (cov_matrix_11 \ (ys - mu1))
    conditional_cov_matrix = cov_matrix_22 - cov_matrix_21 * (cov_matrix_11 \ cov_matrix_12)
    conditional_cov_matrix = 0.5 * conditional_cov_matrix + 0.5 * conditional_cov_matrix'
    new_ys = mvnormal(conditional_mu, conditional_cov_matrix)
    return new_ys
end



const CONSTANT = 1 # 0.2
const LINEAR = 2 # 0.2
const SQUARED_EXP = 3 # 0.2
const PERIODIC = 4 # 0.2
const PLUS = 5 # binary 0.1
const TIMES = 6 # binary 0.1

const node_dist = Float64[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]

const node_type_to_num_children = Dict(
    CONSTANT => 0,
    LINEAR => 0,
    SQUARED_EXP => 0,
    PERIODIC => 0,
    PLUS => 2,
    TIMES => 2)

const max_branch = 2



###########################################
# tree for generating covariance function #
###########################################

# 2) use Tree when computing the AST (likelihood is still non-incremental)
#       - compute the abstract covariance function using the Tree (aggregation functions return covariance function nodes)
#       - evaluate the covariance function to get the covariance matrix

# 3) use Tree when computing the covariance matrix (likelihood is still non-incremental)
#       - compute the covariance matrix using the Tree (aggregation function takes xs, and covariance matrix of children)
#       - evaluate the covariance function to get the covariance matrix


# tree

# production kernel
# input type Nothing (U=Nothing)
# argdiff type: Union{NoArgDiff,Nothing} (DU=Nothing)
# return type: Tuple{Int,Vector{Nothing}} (V=Int,U=Nothing)
# retdiff type: TreeProductionRetDiff{NodeTypeDiff,Nothing}

# aggregation kernel
# input type: Tuple{Int,Vector{Node}} (V=Int,W=Node)
# argdiff type:TreeAggregationArgDiff{NodeTypeDiff,CovNodeDiff} (DV=NodeTypeDiff,DW=CovNodeDiff)
# return type: Node (W=Node)
# retdiff type: CovNodeDiff

# U = Nothing; DU = Nothing
# V = Int; DV = NodeTypeDiff
# W = Node; DW = CovNodeDiff

"""
Indicates whether the node type may have changed or not.
"""
struct NodeTypeDiff
    node_type_same::Bool
end
Gen.isnodiff(node_type_diff::NodeTypeDiff) = node_type_diff.node_type_same

"""
Indicates whether the covariance function may have changed or not.
"""
struct CovNodeDiff
    issame::Bool
end
Gen.isnodiff(cov_node_diff::CovNodeDiff) = cov_node_diff.issame

@gen function production_kernel(_::Nothing)
    node_type = @addr(categorical(node_dist), :type)
    num_children = node_type_to_num_children[node_type]

    @diff begin
        #@assert @argdiff() === noargdiff
        #type_diff = @choicediff(:type)
        #@assert !isnew(type_diff) # always sampled
        #if isnodiff(type_diff)
            #dv = NodeTypeDiff(true)
        #else
            #dv = NodeTypeDiff(prev(type_diff) == node_type)
        #end
        #@retdiff(TreeProductionRetDiff{NodeTypeDiff,Nothing}(dv,Dict{Int,Nothing}()))
        @retdiff(TreeProductionRetDiff{NodeTypeDiff,Nothing}(NodeTypeDiff(false),Dict{Int,Nothing}()))
    end

    return (node_type, [nothing for _=1:num_children])
end


@gen function aggregation_kernel(node_type::Int, children::Vector{Node})
    local node::Node

    # constant kernel
    if node_type == CONSTANT
        @assert length(children) == 0
        param = @addr(uniform_continuous(0, 1), :param) # TODO change prior?
        node = Constant(param)

    # linear kernel
    elseif node_type == LINEAR
        @assert length(children) == 0
        param = @addr(uniform_continuous(0, 1), :param) # TODO change prior?
        node = Linear(param)

    # squared exponential kernel
    elseif node_type == SQUARED_EXP
        @assert length(children) == 0
        length_scale = 0.01 + @addr(uniform_continuous(0, 1), :length_scale) # TODO change prior?
        node = SquaredExponential(length_scale)

    # periodic kernel
    elseif node_type == PERIODIC
        @assert length(children) == 0
        scale = 0.01 + @addr(uniform_continuous(0, 1), :scale) # TODO change prior?
        period = 0.01 + @addr(uniform_continuous(0, 1), :period) # TODO change prior?
        node = Periodic(scale, period)

    # plus combinator
    elseif node_type == PLUS
        @assert length(children) == 2
        node = Plus(children[1], children[2])

    # times combinator
    elseif node_type == TIMES
        @assert length(children) == 2
        node = Times(children[1], children[2])

    # unknown
    else
        error("unknown node type $node_type")
    end

    @diff begin
        #argdiff::TreeAggregationArgDiff{NodeTypeDiff,CovNodeDiff} = @argdiff()
        #@assert !(isa(argdiff.dv, NodeTypeDiff) && isnodiff(argdiff.dv))
        #issame = (
            ## node type is the same
            #(argdiff.dv === noargdiff)
#
            ## all children nodes are the same (isnodiff(dw) for all children)
            #&& length(argdiff.dws) == 0
#
            ## parameters have not changed
            #&& ((node_type == PLUS) ||
                #(node_type == TIMES) ||
                #(node_type == CONSTANT
                    #&& isnodiff(@choicediff(:param))) ||
                #(node_type == LINEAR
                    #&& isnodiff(@choicediff(:param))) ||
                #(node_type == SQUARED_EXP
                    #&& isnodiff(@choicediff(:length_scale))) ||
                #(node_type == PERIODIC
                    #&& isnodiff(@choicediff(:scale))
                    #&& isnodiff(@choicediff(:period)))))

        #@retdiff(CovNodeDiff(issame)) # TODO this was causing bugs..
        @retdiff(CovNodeDiff(false))
   end
    
    return node
end

const cov_fn_generator = Tree(production_kernel, aggregation_kernel, max_branch,
                              Nothing, Int, Node,
                              NodeTypeDiff, Nothing, CovNodeDiff)

@gen function model(xs::Vector{Float64})
    n = length(xs)

    # sample covariance function
    covariance_fn::Node = @addr(cov_fn_generator(nothing, 1), :tree, noargdiff)
    #$println(covariance_fn)

    # sample diagonal noise
    noise = @addr(gamma(1, 1), :noise) + 0.01

    # compute covariance matrix
    cov_matrix = compute_cov_matrix_vectorized(covariance_fn, noise, xs)

    # sample from multivariate normal   
    @addr(mvnormal(zeros(n), cov_matrix), :ys)

    return covariance_fn
end

function compute_log_likelihood(covariance_fn, noise, xs, ys)
    n = length(xs)
    cov_matrix = compute_cov_matrix_vectorized(covariance_fn, noise, xs)
    logpdf(mvnormal, ys, zeros(n), cov_matrix)
end


#########################################
# tree for generating covariance matrix #
#########################################



##############################
# MCMC inference (version 1) #
##############################

@gen function covariance_proposal(prev_trace, root::Int)
    @addr(cov_fn_generator(nothing, root), :tree, noargdiff)
end

function do_inference(xs, ys, num_iters::Int)
    constraints = DynamicAssignment()
    constraints[:ys] = ys
    (trace, _) = generate(model, (xs,), constraints)
    local covariance_fn::Node
    local noise::Float64
    for iter=1:num_iters
        # pick a node to expand
        covariance_fn = get_call_record(trace).retval
        root = pick_random_node(covariance_fn, 1, max_branch)
        trace = mh(model, covariance_proposal, (root,), trace)
        trace = mh(model, noise_proposal, (), trace)
        assignment = get_assignment(trace)
        noise = assignment[:noise]
        log_likelihood = compute_log_likelihood(covariance_fn, noise, xs, ys)
        num_nodes = size(covariance_fn)
        #println("iter: $iter, lik: $log_likelihood, num_nodes: $num_nodes, noise: $noise")
    end
    return (covariance_fn, noise)
end


#########################
# sample some data sets #
#########################

xs = collect(range(0, stop=1, length=50))
Random.seed!(0)

#figure(figsize=(16,16))
#for i=1:16
    #trace = simulate(model, (xs,))
    #ys = get_assignment(trace)[:ys]
    #subplot(4, 4, i)
    #plot(xs, ys)
    #gca()[:set_xlim]((0, 1))
    #gca()[:set_ylim]((-3, 3))
#end
#savefig("data.png")


##########################
# inference experiment 1 #
##########################

function do_inference_experiment_1()
    new_xs = collect(range(0, stop=1.5, length=200))
    figure(figsize=(32,32))
    for i=1:16
        subplot(4, 4, i)
        tic()
        (covariance_fn, noise) = do_inference(xs, ys, 1000)
        toc()
        new_ys = predict_ys(covariance_fn, noise, xs, ys, new_xs)
        plot(xs, ys, color="black")
        plot(new_xs, new_ys, color="red")
        gca()[:set_xlim]((0, 1.5))
        gca()[:set_ylim]((-3, 3))
    end
    savefig("inference.png")
end

#do_inference_experiment_1()

##########################
# inference experiment 2 #
##########################

function do_inference_experiment_2()
    new_xs = collect(range(0, stop=1.5, length=200))
    figure(figsize=(32,32))
    for i=1:16
        subplot(4, 4, i)
        tic()
        (covariance_fn, noise) = do_inference2(xs, ys, 1000)
        toc()
        new_ys = predict_ys(covariance_fn, noise, xs, ys, new_xs)
        plot(xs, ys, color="black")
        plot(new_xs, new_ys, color="red")
        gca()[:set_xlim]((0, 1.5))
        gca()[:set_ylim]((-3, 3))
    end
    savefig("inference2.png")
end

do_inference_experiment_2()










# backpropagation
# to support backpropagation of the likelihood with respect to real-valued
# parameters in the covariance function, we will need to invent a data
# structure to store the gradient with respect to the covariance function. this
# could be a tree-structured object that closely mirrors the tree of the
# covariance function object itself. the GP function and the aggregation kernel
# module will both need to know about this specialized gradient object (the GP
# function will produce it as a return value from backprop, and the aggregation
# kernel function will accept it as an input to backprop)
