abstract type StaticIRNode end

struct TrainableParameterNode <: StaticIRNode
    name::Symbol
    typ::Union{Symbol,Expr,QuoteNode}
end

struct ArgumentNode <: StaticIRNode
    name::Symbol
    typ::Union{Symbol,Expr,QuoteNode}
    compute_grad::Bool
end

struct JuliaNode <: StaticIRNode
    fn::Function
    inputs::Vector{StaticIRNode}
    name::Symbol
    typ::Union{Symbol,Expr,QuoteNode}
end

struct GenerativeFunctionCallNode <: StaticIRNode
    generative_function::GenerativeFunction
    inputs::Vector{StaticIRNode}
    addr::Symbol
    name::Symbol
    typ::Union{Symbol,Expr,QuoteNode}
end

struct StaticIR
    nodes::Vector{StaticIRNode}
    trainable_param_nodes::Vector{TrainableParameterNode}
    arg_nodes::Vector{ArgumentNode}
    call_nodes::Vector{GenerativeFunctionCallNode}
    julia_nodes::Vector{JuliaNode}
    return_node::StaticIRNode
    accepts_output_grad::Bool
end

mutable struct StaticIRBuilder
    nodes::Vector{StaticIRNode}
    node_set::Set{StaticIRNode}
    trainable_param_nodes::Vector{TrainableParameterNode}
    arg_nodes::Vector{ArgumentNode}
    call_nodes::Vector{GenerativeFunctionCallNode}
    julia_nodes::Vector{JuliaNode}
    return_node::Union{Nothing,StaticIRNode}
    vars::Set{Symbol}
    addrs_to_call_nodes::Dict{Symbol,GenerativeFunctionCallNode}
    accepts_output_grad::Bool
end

function StaticIRBuilder()
    nodes = Vector{StaticIRNode}()
    node_set = Set{StaticIRNode}()
    trainable_param_nodes = Vector{TrainableParameterNode}()
    arg_nodes = Vector{ArgumentNode}()
    call_nodes = Vector{GenerativeFunctionCallNode}()
    julia_nodes = Vector{JuliaNode}()
    return_node = nothing
    vars = Set{Symbol}()
    addrs_to_call_nodes = Dict{Symbol,GenerativeFunctionCallNode}()
    accepts_output_grad = false
    StaticIRBuilder(nodes, node_set, trainable_param_nodes, arg_nodes, call_nodes,
        julia_nodes,
        return_node, vars, addrs_to_call_nodes,
        accepts_output_grad)
end

function build_ir(builder::StaticIRBuilder)
    if builder.return_node === nothing
        builder.return_node = add_constant_node!(builder, nothing)
    end
    StaticIR(
        builder.nodes,
        builder.trainable_param_nodes,
        builder.arg_nodes,
        builder.call_nodes,
        builder.julia_nodes,
        builder.return_node,
        builder.accepts_output_grad)
end

function check_unique_var(builder::StaticIRBuilder, name::Symbol)
    if name in builder.vars
        error("Variable name $name is not unique")
    end
end

function check_inputs_exist(builder::StaticIRBuilder, input_nodes)
    for input_node in input_nodes
        if !(input_node in builder.node_set)
            error("Node $input_node was not previously added to the IR")
        end
    end
end

function check_addr_unique(builder::StaticIRBuilder, addr::Symbol)
    if haskey(builder.addrs_to_call_nodes, addr)
        error("Address $addr was not unique")
    end
end

function _add_node!(builder::StaticIRBuilder, node::StaticIRNode)
    push!(builder.nodes, node)
    push!(builder.node_set, node)
end

function add_trainable_param_node!(builder::StaticIRBuilder,
                                   name::Symbol;
                                   typ::Union{Symbol,Expr,QuoteNode}=QuoteNode(Any))
    check_unique_var(builder, name)
    node = TrainableParameterNode(name, typ)
    _add_node!(builder, node)
    push!(builder.trainable_param_nodes, node)
    node
end


function add_argument_node!(builder::StaticIRBuilder;
                            name::Symbol=gensym(),
                            typ::Union{Symbol,Expr,QuoteNode}=QuoteNode(Any),
                            compute_grad=false)
    check_unique_var(builder, name)
    node = ArgumentNode(name, typ, compute_grad)
    _add_node!(builder, node)
    push!(builder.arg_nodes, node)
    node
end

function add_julia_node!(builder::StaticIRBuilder, fn::Function;
                         inputs::Vector=[],
                         name::Symbol=gensym(),
                         typ::Union{Symbol,Expr,QuoteNode}=QuoteNode(Any))
    check_unique_var(builder, name)
    check_inputs_exist(builder, inputs)
    node = JuliaNode(fn, inputs, name, typ)
    _add_node!(builder, node)
    push!(builder.julia_nodes, node)
    node
end

function add_constant_node!(builder::StaticIRBuilder, val,
        name::Symbol=gensym(),
        typ::Union{Symbol,Expr,QuoteNode}=QuoteNode(typeof(val)))
    check_unique_var(builder, name)
    # NOTE: not wrapping it in a Diffed means it is interpreted as a constant
    node = JuliaNode(() -> val, [], name, typ)
    _add_node!(builder, node)
    push!(builder.julia_nodes, node)
    node
end

function add_addr_node!(builder::StaticIRBuilder, gen_fn::GenerativeFunction;
                        inputs::Vector=[], addr::Symbol=gensym(),
                        name::Symbol=gensym())
    check_unique_var(builder, name)
    check_addr_unique(builder, addr)
    check_inputs_exist(builder, inputs)
    typ = QuoteNode(get_return_type(gen_fn))
    node = GenerativeFunctionCallNode(gen_fn, inputs, addr, name, typ)
    _add_node!(builder, node)
    builder.addrs_to_call_nodes[addr] = node
    push!(builder.call_nodes, node)
    node
end

function set_return_node!(builder::StaticIRBuilder, node::StaticIRNode)
    if builder.return_node !== nothing
        error("Return node already set")
    end
    builder.return_node = node
    nothing
end

function set_accepts_output_grad!(builder::StaticIRBuilder, value::Bool)
    builder.accepts_output_grad = value
end

export StaticIR, StaticIRBuilder, build_ir
export add_trainable_param_node!
export add_argument_node!
export add_julia_node!
export add_constant_node!
export add_addr_node!
export set_return_node!
