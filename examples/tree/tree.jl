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

# input type U: Nothing
# V = Int (what type of node we are?)
# return type: Tuple{Node,Vector{Nothing}}

@gen function production_kernel(_::Nothing)
    node_type = @addr(categorical(node_dist), :type)
    if node_type == INPUT_NODE
        num_children = 0
    elseif node_type == CONSTANT_NODE
        num_children = 0
    elseif node_type == PLUS_NODE
        num_children = 2
    elseif node_type == MINUS_NODE
        num_children = 2
    elseif node_type == TIMES_NODE
        num_children = 2
    elseif node_type == CHANGE_NODE
        num_children = 2
    else
        error("unknown node type $node_type")
    end
    return (node_type, Nothing[nothing for _=1:num_children])
end

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
    return node
end

tree = Tree(production_kernel, aggregation_kernel, Nothing, Int, Node, 2)

@gen function model()
    root_node::Node = @addr(tree(nothing), :tree)
end

for i=1:100
    trace = simulate(model, ())
    println(get_assignment(trace))
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
