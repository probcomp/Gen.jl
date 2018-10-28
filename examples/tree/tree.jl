using Gen

#################################
# covariance function AST nodes #
#################################

abstract type Node end

struct InputSymbolNode <: Node
end
eval_ast(node::InputSymbolNode, x::Float64) = x
size(::InputSymbolNode) = 1

struct ConstantNode{T} <: Node
    value::T
end
eval_ast(node::ConstantNode, x::Float64) = node.value
size(::ConstantNode) = 1

abstract type BinaryOpNode <: Node end
function eval_ast(node::BinaryOpNode, x::Float64)
    eval_op(node, eval_ast(node.left, x), eval_ast(node.right, x))
end
size(node::BinaryOpNode) = node.size

struct ChangepointNode <: BinaryOpNode
    loc::Float64
    left::Node
    right::Node
    size::Int
end
ChangepointNode(left, right, loc) = ChangepointNode(loc, left, right, size(left) + size(right) + 1)
function eval_ast(node::ChangepointNode, x::Float64)
    if x < node.loc
        eval_ast(node.left, x)
    else
        eval_ast(node.right, x)
    end
end
    
struct PlusNode <: BinaryOpNode
    left::Node
    right::Node
    size::Int
end
PlusNode(left, right) = PlusNode(left, right, size(left) + size(right) + 1)
eval_op(::PlusNode, a, b) = a + b

struct MinusNode <: BinaryOpNode
    left::Node
    right::Node
    size::Int
end
MinusNode(left, right) = MinusNode(left, right, size(left) + size(right) + 1)
eval_op(::MinusNode, a, b) = a - b

struct TimesNode <: BinaryOpNode
    left::Node
    right::Node
    size::Int
end
TimesNode(left, right) = TimesNode(left, right, size(left) + size(right) + 1)
eval_op(::TimesNode, a, b) = a * b

#########
# model #
#########

const INPUT_NODE = 1
const CONSTANT_NODE = 2
const PLUS_NODE = 3
const MINUS_NODE = 4
const TIMES_NODE = 5
const CHANGE_NODE = 6

const node_dist = Float64[0.4, 0.4, 0.2/4, 0.2/4, 0.2/4, 0.2/4]

const node_type_to_num_children = Dict(
    INPUT_NODE => 0,
    CONSTANT_NODE => 0,
    PLUS_NODE => 2,
    MINUS_NODE => 2,
    TIMES_NODE => 2,
    CHANGE_NODE => 2)



##########################################
# data types for incremental computation #
##########################################

# no information is passed from the parent to each child
const U = Nothing
const u = nothing
const DU = Nothing

"""
Data type indicating whether the node type changed or not.
"""
struct DV
    node_type_same::Bool
end
Gen.isnodiff(dv::DV) = dv.node_type_same

"""
Indicates whether the covariance function may have changed or not.
"""
struct DW
    issame::Bool
end
Gen.isnodiff(dw::DW) = dw.issame


#####################
# production kernel #
#####################

# input type Nothing (U=Nothing)
# argdiff type: Nothing (DU=Nothing)
# return type: Tuple{Int,Vector{Nothing}} (V=Int,U=Nothing)
# retdiff type: 

@gen function production_kernel(_::Nothing)
    node_type = @addr(categorical(node_dist), :type)
    num_children = node_type_to_num_children[node_type]

    # compute retdiff
    # none of the arguments to existing children change, because arguments are always nothing
    @diff begin
        @assert @argdiff() === noargdiff
        @assert !isnew(@choicediff(:type)) # always sampled
        if isnodiff(@choicediff(:type))
            dv = DV(true)
        else
            dv = DV(prev(@choicediff(:type)) == node_type)
        end
        @retdiff(TreeProductionRetDiff{DV,DU}(dv,Dict{Int,DU}()))
    end

    return (node_type, U[u for _=1:num_children])
end


######################
# aggregation kernel #
######################

# input type: Tuple{Int,Vector{Node}} (V=Int,W=Node)
# argdiff type:TreeAggregationArgDiff{Nothing,Nothing} (DV=Nothing,DW=Nothing)
# return type: Node (W=Node)
# retdiff type: Nothing (DW)

@gen function aggregation_kernel(node_type::Int, children_inputs::Vector{Node})
    local node::Node
    if node_type == INPUT_NODE
        @assert length(children_inputs) == 0
        node = InputSymbolNode()
    elseif node_type == CONSTANT_NODE
        @assert length(children_inputs) == 0
        param = @addr(normal(0, 3), :const)
        node = ConstantNode(param)
    elseif node_type == PLUS_NODE
        @assert length(children_inputs) == 2
        node = PlusNode(children_inputs[1], children_inputs[2])
    elseif node_type == MINUS_NODE
        @assert length(children_inputs) == 2
        node = MinusNode(children_inputs[1], children_inputs[2])
    elseif node_type == TIMES_NODE
        @assert length(children_inputs) == 2
        node = TimesNode(children_inputs[1], children_inputs[2])
    elseif node_type == CHANGE_NODE
        @assert length(children_inputs) == 2
        loc = @addr(normal(0, 3), :changept)
        node = ChangepointNode(children_inputs[1], children_inputs[2], loc)
    else
        error("unknown node type $node_type")
    end

    @diff begin
        argdiff::TreeAggregationArgDiff{DV,DW} = @argdiff()
        issame = (
            # node type is the same
            isnodiff(argdiff.dv)

            # all children nodes are the same (isnodiff(dw) for all children)
            && length(argdiff.dws) == 0

            # parameters have not changed
            && ((node_type == CONSTANT_NODE && isnodiff(@choicediff(:const))) ||
                (node_type == CHANGE_NODE && isnodiff(@choicediff(:changept)))))
        @retdiff(DW(issame))
   end
    
    return node
end

# U = Nothing; DU = Nothing
# V = Int; DV = Nothing
# W = Node; DW = Nothing
tree = Tree(production_kernel, aggregation_kernel, 2, U, Int, Node, DV, DU, DW)

## test generate ##

model_expr = :(@gen function model()
    root_node::Node = @addr(tree(u, 1), :tree, noargdiff)
end)

println(macroexpand(model_expr))
eval(model_expr)

for i=1:100
    trace = simulate(model, ())
    println(get_assignment(trace))
end

## test update ##

@gen function proposal(root::Int)
    @addr(tree(u, root), :tree, noargdiff)
end

for i=1:100
    trace = simulate(model, ())
    
    println("\nprevious trace:")
    println(get_assignment(trace))
    
    
    proposal_trace = simulate(proposal, (1,))
    
    println("\nproposal trace:")
    println(get_assignment(proposal_trace))
    
    (new_trace, weight, discard, retdiff) = update(model, (),
            noargdiff, trace, get_assignment(proposal_trace))
    
    println("\nnew trace:")
    println(get_assignment(new_trace))
end

# TODO backpropagation

# to support backpropagation of the likelihood with respect to real-valued
# parameters in the covariance function, we will need to invent a data
# structure to store the gradient with respect to the covariance function. this
# could be a tree-structured object that closely mirrors the tree of the
# covariance function object itself. the GP function and the aggregation kernel
# module will both need to know about this specialized gradient object (the GP
# function will produce it as a return value from backprop, and the aggregation
# kernel function will accept it as an input to backprop)
