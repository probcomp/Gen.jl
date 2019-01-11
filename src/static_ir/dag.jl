abstract type StaticIRNode end
abstract type RegularNode <: StaticIRNode end
abstract type DiffNode <: StaticIRNode end

## regular nodes ##

struct ArgumentNode <: RegularNode
    name::Symbol
    typ::Type
    compute_grad::Bool
end

struct JuliaNode <: RegularNode
    fn::Function
    inputs::Vector{RegularNode}
    name::Symbol
    typ::Type
end

struct RandomChoiceNode <: RegularNode
    dist::Distribution
    inputs::Vector{RegularNode}
    addr::Symbol
    name::Symbol
    typ::Type
end

struct GenerativeFunctionCallNode <: RegularNode
    generative_function::GenerativeFunction
    inputs::Vector{RegularNode}
    addr::Symbol
    argdiff::StaticIRNode
    name::Symbol
    typ::Type
end

## diff nodes ##

struct DiffJuliaNode <: DiffNode
    fn::Function
    inputs::Vector{StaticIRNode}
    name::Symbol
    typ::Type
end

struct ReceivedArgDiffNode <: DiffNode
    name::Symbol
    typ::Type

    # syntax:
    # @diff x::T = @argdiff()

    # defaults to declared_type = Any
    # @diff _::Any = @argdiff()
end

struct ChoiceDiffNode <: DiffNode
    choice_node::RandomChoiceNode
    name::Symbol
    typ::Type
end

struct CallDiffNode <: DiffNode
    call_node::GenerativeFunctionCallNode
    name::Symbol
    typ::Type
end

struct StaticIR
    nodes::Vector{StaticIRNode}
    arg_nodes::Vector{ArgumentNode}
    choice_nodes::Vector{RandomChoiceNode}
    call_nodes::Vector{GenerativeFunctionCallNode}
    return_node::RegularNode
    retdiff_node::StaticIRNode
    received_argdiff_node::ReceivedArgDiffNode
    accepts_output_grad::Bool
end

mutable struct StaticIRBuilder
    nodes::Vector{StaticIRNode}
    node_set::Set{StaticIRNode}
    arg_nodes::Vector{ArgumentNode}
    choice_nodes::Vector{RandomChoiceNode}
    call_nodes::Vector{GenerativeFunctionCallNode}
    return_node::Union{Nothing,RegularNode}
    retdiff_node::Union{Nothing,StaticIRNode}
    received_argdiff_node::Union{Nothing,ReceivedArgDiffNode}
    vars::Set{Symbol}
    addrs_to_choice_nodes::Dict{Symbol,RandomChoiceNode}
    addrs_to_call_nodes::Dict{Symbol,GenerativeFunctionCallNode}
    accepts_output_grad::Bool
end

function StaticIRBuilder()
    nodes = Vector{StaticIRNode}()
    node_set = Set{StaticIRNode}()
    arg_nodes = Vector{ArgumentNode}()
    choice_nodes = Vector{RandomChoiceNode}()
    call_nodes = Vector{GenerativeFunctionCallNode}()
    return_node = nothing
    retdiff_node = nothing
    received_argdiff_node = nothing
    vars = Set{Symbol}()
    addrs_to_choice_nodes = Dict{Symbol,RandomChoiceNode}()
    addrs_to_call_nodes = Dict{Symbol,GenerativeFunctionCallNode}()
    accepts_output_grad = false
    StaticIRBuilder(nodes, node_set, arg_nodes, choice_nodes, call_nodes,
        return_node, retdiff_node,
        received_argdiff_node, vars, addrs_to_choice_nodes, addrs_to_call_nodes,
        accepts_output_grad)
end

function build_ir(builder::StaticIRBuilder)
    if builder.received_argdiff_node === nothing
        builder.received_argdiff_node = add_received_argdiff_node!(builder)
    end
    if builder.return_node === nothing
        builder.return_node = add_constant_node!(builder, nothing)
    end
    if builder.retdiff_node === nothing
        builder.retdiff_node = add_constant_node!(builder, DefaultRetDiff())
    end
    StaticIR(
        builder.nodes,
        builder.arg_nodes,
        builder.choice_nodes,
        builder.call_nodes,
        builder.return_node,
        builder.retdiff_node,
        builder.received_argdiff_node,
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
    if haskey(builder.addrs_to_choice_nodes, addr) || haskey(builder.addrs_to_call_nodes, addr)
        error("Address $addr was not unique")
    end
end

function _add_node!(builder::StaticIRBuilder, node::StaticIRNode)
    push!(builder.nodes, node)
    push!(builder.node_set, node)
end

function add_argument_node!(builder::StaticIRBuilder;
                            name::Symbol=gensym(), typ::Type=Any,
                            compute_grad=false)
    check_unique_var(builder, name)
    node = ArgumentNode(name, typ, compute_grad)
    _add_node!(builder, node)
    push!(builder.arg_nodes, node)
    node
end

function add_julia_node!(builder::StaticIRBuilder, fn::Function;
                         inputs::Vector=[],
                         name::Symbol=gensym(), typ::Type=Any)
    check_unique_var(builder, name)
    check_inputs_exist(builder, inputs)
    node = JuliaNode(fn, inputs, name, typ)
    _add_node!(builder, node)
    node
end

function add_constant_node!(builder::StaticIRBuilder, val::T, name::Symbol=gensym()) where {T}
    check_unique_var(builder, name)
    node = JuliaNode(() -> val, [], name, T)
    _add_node!(builder, node)
    node
end

function add_constant_diff_node!(builder::StaticIRBuilder, val::T, name::Symbol=gensym()) where {T}
    check_unique_var(builder, name)
    node = DiffJuliaNode(() -> val, [], name, T)
    _add_node!(builder, node)
    node
end

function add_addr_node!(builder::StaticIRBuilder, dist::Distribution;
                        inputs::Vector=[], addr::Symbol=gensym(),
                        name::Symbol=gensym(), typ::Type=Any)
    check_unique_var(builder, name)
    check_addr_unique(builder, addr)
    check_inputs_exist(builder, inputs)
    node = RandomChoiceNode(dist, inputs, addr, name, typ)
    _add_node!(builder, node)
    builder.addrs_to_choice_nodes[addr] = node
    push!(builder.choice_nodes, node)
    node
end

function add_addr_node!(builder::StaticIRBuilder, gen_fn::GenerativeFunction;
                        inputs::Vector=[], addr::Symbol=gensym(),
                        argdiff::StaticIRNode=add_constant_node!(builder, unknownargdiff),
                        name::Symbol=gensym(), typ::Type=Any)
    check_unique_var(builder, name)
    check_addr_unique(builder, addr)
    check_inputs_exist(builder, inputs)
    if !(argdiff in builder.node_set)
        error("Node $argdiff was not previously added to the IR")
    end
    node = GenerativeFunctionCallNode(gen_fn, inputs, addr, argdiff, name, typ)
    _add_node!(builder, node)
    builder.addrs_to_call_nodes[addr] = node
    push!(builder.call_nodes, node)
    node
end

function add_diff_julia_node!(builder::StaticIRBuilder, fn::Function;
                              inputs::Vector=[], name::Symbol=gensym(), typ::Type=Any)
    check_unique_var(builder, name)
    check_inputs_exist(builder, inputs)
    node = DiffJuliaNode(fn, inputs, name, typ)
    _add_node!(builder, node)
    node
end

function add_received_argdiff_node!(builder::StaticIRBuilder;
                                    name::Symbol=gensym(), typ::Type=Any)
    check_unique_var(builder, name)
    if builder.received_argdiff_node !== nothing
        error("A received argdiff node was already added")
    end
    node = ReceivedArgDiffNode(name, typ)
    _add_node!(builder, node)
    builder.received_argdiff_node = node
    node
end

function add_choicediff_node!(builder::StaticIRBuilder, addr::Symbol;
                              name::Symbol=gensym(), typ::Type=Any)
    check_unique_var(builder, name)
    choice_node = builder.addrs_to_choice_nodes[addr]
    node = ChoiceDiffNode(choice_node, name, typ)
    _add_node!(builder, node)
    node
end

function add_calldiff_node!(builder::StaticIRBuilder, addr::Symbol;
                            name::Symbol=gensym(), typ::Type=Any)
    check_unique_var(builder, name)
    call_node = builder.addrs_to_call_nodes[addr]
    node = CallDiffNode(call_node, name, typ)
    _add_node!(builder, node)
    node
end

function set_return_node!(builder::StaticIRBuilder, node::RegularNode)
    if builder.return_node !== nothing
        error("Return node already set")
    end
    builder.return_node = node
    nothing
end

function set_retdiff_node!(builder::StaticIRBuilder, node::StaticIRNode)
    if builder.retdiff_node !== nothing
        error("Retdiff node already set")
    end
    builder.retdiff_node = node
    nothing
end

function set_accepts_output_grad!(builder::StaticIRBuilder, value::Bool)
    builder.accepts_output_grad = value
end

export StaticIR, StaticIRBuilder, build_ir
export add_argument_node!
export add_julia_node!
export add_constant_node!
export add_constant_diff_node!
export add_addr_node!
export add_received_argdiff_node!
export add_choicediff_node!
export add_calldiff_node!
export add_diff_julia_node!
export set_retdiff_node!
export set_return_node!
