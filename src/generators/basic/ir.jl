using PyCall
@pyimport graphviz as gv

struct ParamInfo
    name::Symbol
    typ::Any
end

abstract type Node end
abstract type ExprNode <: Node end
abstract type ValueNode <: Node end

# utility

function replace_input_symbols(symbols::Set{Symbol}, value, trace::Symbol)
    value
end

function replace_input_symbols(symbols::Set{Symbol}, expr::Expr, trace::Symbol)
    args = [replace_input_symbols(symbols, arg, trace) for arg in expr.args]
    Expr(expr.head, args...)
end

function replace_input_symbols(symbols::Set{Symbol}, symbol::Symbol, trace::Symbol)
    if symbol in symbols
        return Expr(:(.), trace, QuoteNode(value_field(symbol)))
    else
        return symbol
    end
end

# statement nodes

struct ArgsChangeNode <: ExprNode
    output::ValueNode
end

function Base.print(io::IO, node::ArgsChangeNode)
    write(io, "@argschange")
end

parents(::ArgsChangeNode) = []

struct ReadNode <: ExprNode
    input_nodes::Vector{ValueNode}
    output::ValueNode
    address::Expr
end

function Base.print(io::IO, node::ReadNode)
    write(io, "@read $(node.address)")
end

function expr_read_from_trace(node::ReadNode, trace::Symbol)
    input_symbols = Set{Symbol}([node.name for node in node.input_nodes])
    replace_input_symbols(input_symbols, node.address, trace)
end

parents(node::ReadNode) = node.input_nodes

struct AddrDistNode{T} <: ExprNode
    input_nodes::Vector{ValueNode}
    output::Nullable{ValueNode}
    dist::Distribution{T}
    address::Symbol
end

function Base.print(io::IO, node::AddrDistNode)
    write(io, "@addr $(node.dist) $(node.address)")
end

parents(node::AddrDistNode) = node.input_nodes

struct AddrGeneratorNode{T,U} <: ExprNode
    input_nodes::Vector{ValueNode}
    output::Nullable{ValueNode} # TODO
    gen::Generator{T,U}
    address::Symbol
    change_node::ValueNode
end

function Base.print(io::IO, node::AddrGeneratorNode)
    write(io, "@addr $(node.gen) $(node.address)")
end

parents(node::AddrGeneratorNode) = vcat([node.change_node], node.input_nodes)

struct AddrChangeNode <: ExprNode
    address::Symbol
    addr_node::Union{AddrDistNode,AddrGeneratorNode}
    output::ValueNode
end

function Base.print(io::IO, node::AddrChangeNode)
    write(io, "@change $(node.address)")
end

# so that the @change will be placed after its @addr node
parents(node::AddrChangeNode) = [node.addr_node]

struct JuliaNode <: ExprNode
    input_nodes::Vector{ValueNode}
    output::ValueNode
    expr_or_value::Any
end

function Base.print(io::IO, node::JuliaNode)
    write(io, "Julia $(gensym())")
end

parents(node::JuliaNode) = node.input_nodes

function expr_read_from_trace(node::JuliaNode, trace::Symbol)
    input_symbols = Set{Symbol}([node.name for node in node.input_nodes])
    replace_input_symbols(input_symbols, node.expr_or_value, trace)
end


# value nodes
struct ArgumentValueNode <: ValueNode
    name::Symbol
    typ::Any
end

function Base.print(io::IO, node::ArgumentValueNode)
    write(io, "Arg $(node.name)")
end

parents(node::ArgumentValueNode) = []
get_type(node::ArgumentValueNode) = node.typ

mutable struct ExprValueNode <: ValueNode
    name::Symbol
    typ::Any
    finished::Bool
    source::ExprNode
    function ExprValueNode(name, typ)
        new(name, typ, false)
    end
end

function finish!(node::ExprValueNode, source::ExprNode)
    @assert !node.finished
    node.finished = true
    node.source = source
end

function Base.print(io::IO, node::ExprValueNode)
    write(io, "$(node.name)  $(node.typ)")
end

parents(node::ExprValueNode) = [node.source]
get_type(node::ExprValueNode) = node.typ

struct ParamValueNode <: ValueNode
    param::ParamInfo
end

function Base.print(io::IO, node::ParamValueNode)
    write(io, "Param $(node.param.name)  $(node.param.typ)")
end

parents(node::ParamValueNode) = []
get_type(node::ParamValueNode) = node.param.typ

mutable struct BasicBlockIR
    arg_nodes::Vector{ArgumentValueNode}
    params::Vector{ParamInfo}
    value_nodes::Dict{Symbol, ValueNode}
    addr_dist_nodes::Dict{Symbol, AddrDistNode}
    addr_gen_nodes::Dict{Symbol, AddrGeneratorNode}
    all_nodes::Set{Node}
    output_node::Nullable{ValueNode}
    retchange_node::Nullable{ValueNode}
    args_change_node::Nullable{ArgsChangeNode}
    addr_change_nodes::Set{AddrChangeNode}
    generator_input_change_nodes::Set{ValueNode}
    finished::Bool

    # nodes that depend on incremental values (@argchange() and @change())
    # that aren't part of the actual function. used to check that there is no
    # data flow from incremental values to the actual function's observable
    # behavior (trace or return value).
    incremental_nodes::Set{ValueNode}

    expr_nodes_sorted::Vector{ExprNode}
    function BasicBlockIR()
        arg_nodes = Vector{ArgumentValueNode}()
        params = Vector{ParamInfo}()
        value_nodes = Dict{Symbol,ValueNode}()
        addr_dist_nodes = Dict{Symbol, AddrDistNode}()
        addr_gen_nodes = Dict{Symbol, AddrGeneratorNode}()
        all_nodes = Set{Node}()
        output_node = Nullable{ValueNode}()
        retchange_node = Nullable{ValueNode}()
        args_change_node = Nullable{ArgsChangeNode}()
        addr_change_nodes = Set{AddrChangeNode}()
        generator_input_change_nodes = Set{ValueNode}()
        finished = false
        incremental_nodes = Set{ValueNode}()
        new(arg_nodes, params, value_nodes, addr_dist_nodes,
                     addr_gen_nodes, all_nodes, output_node, retchange_node,
                     args_change_node, addr_change_nodes,
                     generator_input_change_nodes,
                     finished, incremental_nodes)
    end
end

function toposort_visit!(node, permanent_marked, temporary_marked,
                         unmarked, reverse_list)
    if node in permanent_marked
        return
    end
    if node in temporary_marked
        error("cycle found")
    end
    push!(temporary_marked, node)
    for parent in parents(node)
        toposort_visit!(parent, permanent_marked, temporary_marked, unmarked, reverse_list)
    end
    push!(permanent_marked, node)
    push!(reverse_list, node)
end

function toposort_expr_nodes(nodes)
    permanent_marked = Set{Node}()
    temporary_marked = Set{Node}()
    unmarked = Set{Node}(nodes)
    reverse_list = Vector{Node}()
    while !isempty(unmarked)
        node = pop!(unmarked)
        toposort_visit!(node, permanent_marked, temporary_marked, unmarked, reverse_list)
    end
    priorities = Dict(node => i for (i, node) in enumerate(reverse_list))
    expr_nodes = collect(
        Iterators.filter((node) -> isa(node, ExprNode), keys(priorities)))
    convert(Vector{ExprNode}, sort(expr_nodes, by=(node) -> priorities[node]))
end

function finish!(ir::BasicBlockIR)
    ir.finished = true
    ir.expr_nodes_sorted = toposort_expr_nodes(ir.all_nodes)
end


# NOTE: currently we don't properly support list comprehensions and closures
# and let (anything that defines a new environment)

function render_graph(ir::BasicBlockIR, fname)
    # graphviz
    dot = gv.Digraph() # comment = ?
    nodes_to_name = Dict{Node,String}()
    for node in ir.all_nodes
        io = IOBuffer()
        print(io, node)
        nodes_to_name[node] = String(take!(io))
    end
    for node in ir.all_nodes
        if isa(node, ValueNode)
            shape = "box"
        else
            shape = "ellipse"
        end
        if node in ir.incremental_nodes
            color = "lightblue"
        else
            color = "lightgray"
        end
        if !isnull(ir.output_node) && get(ir.output_node) == node
            @assert !(node in ir.incremental_nodes)
            color = "firebrick1"
        end
        if !isnull(ir.retchange_node) && get(ir.retchange_node) == node
            color = "darkolivegreen2"
        end
        dot[:node](nodes_to_name[node], nodes_to_name[node], shape=shape, color=color, style="filled")
        for parent in parents(node)
            dot[:edge](nodes_to_name[parent], nodes_to_name[node])
        end
    end
    for node in values(ir.addr_gen_nodes)
        dot[:edge](nodes_to_name[node.change_node], nodes_to_name[node], style="dashed")
    end
    dot[:render](fname, view=true)
end

function add_argument!(ir::BasicBlockIR, name::Symbol, typ)
    @assert !ir.finished
    node = ArgumentValueNode(name, typ)
    push!(ir.arg_nodes, node)
    push!(ir.all_nodes, node)
    ir.value_nodes[name] = node
end

function add_param!(ir::BasicBlockIR, name::Symbol, typ)
    @assert !ir.finished
    param_info = ParamInfo(name, typ)
    push!(ir.params, param_info)
    node = ParamValueNode(param_info)
    if haskey(ir.value_nodes, name)
        error("Name $name already used")
    end
    ir.value_nodes[name] = node
    push!(ir.all_nodes, node)
end

function resolve_symbols_set(ir::BasicBlockIR, symbol::Symbol)
    symbols = Set{Symbol}()
    if haskey(ir.value_nodes, symbol)
        push!(symbols, symbol)
    end
    symbols
end

resolve_symbols_set(ir::BasicBlockIR, expr) = Set{Symbol}()

function resolve_symbols_set(ir::BasicBlockIR, expr::Expr)
    symbols = Set{Symbol}()
    if expr.head == :(.)
        union!(symbols, resolve_symbols_set(ir, expr.args[1]))
    elseif expr.head != :line
        for operand in expr.args
            union!(symbols, resolve_symbols_set(ir, operand))
        end
    end
    symbols
end

function _get_input_node!(ir::BasicBlockIR, expr, typ)
    name = gensym()
    value_node = ExprValueNode(name, typ)
    ir.value_nodes[name] = value_node
    push!(ir.all_nodes, value_node)
    input_nodes = map((sym) -> ir.value_nodes[sym],
                      collect(resolve_symbols_set(ir, expr)))
    expr_node = JuliaNode(input_nodes, value_node, expr)
    finish!(value_node, expr_node)
    push!(ir.all_nodes, expr_node)
    if any([node in ir.incremental_nodes for node in input_nodes])
        push!(ir.incremental_nodes, value_node)
    end
    value_node
end

function _get_input_node!(ir::BasicBlockIR, name::Symbol, typ)
    node = ir.value_nodes[name]
    #@assert get_type(node) == typ # TODO
    node
end

function add_read!(ir::BasicBlockIR, expr, typ, name::Symbol)
    @assert !ir.finished
    value_node = ExprValueNode(name, typ)
    input_nodes = map((sym) -> ir.value_nodes[sym],
                      collect(resolve_symbols_set(ir, expr)))
    expr_node = ReadNode(input_nodes, value_node, expr)
    finish!(value_node, expr_node)
    ir.value_nodes[name] = value_node
    push!(ir.all_nodes, expr_node)
    push!(ir.all_nodes, value_node)
    if any([node in ir.incremental_nodes for node in input_nodes])
        push!(ir.incremental_nodes, value_node)
    end
    nothing
end

function incremental_dependency_error(addr)
    error("@addr argument cannot depend on @change or @argschange statements (address $addr)")
end

function add_addr!(ir::BasicBlockIR, addr::Symbol, dist::Distribution{T}, args::Vector, typ, name::Symbol) where {T}
    @assert !ir.finished
    types = get_static_argument_types(dist)
    input_nodes = ValueNode[
        _get_input_node!(ir, expr, typ) for (expr, typ) in zip(args, types)]
    if any([node in ir.incremental_nodes for node in input_nodes])
        incremental_dependency_error(addr)
    end
    value_node = ExprValueNode(name, typ)
    expr_node = AddrDistNode(input_nodes, Nullable{ValueNode}(value_node), dist, addr)
    finish!(value_node, expr_node)
    ir.value_nodes[name] = value_node
    push!(ir.all_nodes, expr_node)
    ir.addr_dist_nodes[addr] = expr_node
    push!(ir.all_nodes, value_node)
    nothing
end

function add_addr!(ir::BasicBlockIR, addr::Symbol, dist::Distribution{T}, args::Vector) where {T}
    @assert !ir.finished
    types = get_static_argument_types(dist)
    input_nodes = ValueNode[
        _get_input_node!(ir, expr, typ) for (expr, typ) in zip(args, types)]
    if any([node in ir.incremental_nodes for node in input_nodes])
        incremental_dependency_error(addr)
    end
    expr_node = AddrDistNode(input_nodes, Nullable{ValueNode}(), dist, addr)
    push!(ir.all_nodes, expr_node)
    ir.addr_dist_nodes[addr] = expr_node
    nothing
end


function add_addr!(ir::BasicBlockIR, addr::Symbol, gen::Generator{T,U}, args::Vector, typ, name::Symbol, change_expr) where {T,U}
    @assert !ir.finished
    types = get_static_argument_types(gen)
    input_nodes = ValueNode[
        _get_input_node!(ir, expr, typ) for (expr, typ) in zip(args, types)]
    if any([node in ir.incremental_nodes for node in input_nodes])
        incremental_dependency_error(addr)
    end
    value_node = ExprValueNode(name, typ)
    change_node = _get_input_node!(ir, change_expr, Expr(:curly, :Nullable, get_change_type(gen)))
    push!(ir.generator_input_change_nodes, change_node)
    expr_node = AddrGeneratorNode(input_nodes, Nullable{ValueNode}(value_node), gen, addr, change_node)
    finish!(value_node, expr_node)
    ir.value_nodes[name] = value_node
    push!(ir.all_nodes, expr_node)
    ir.addr_gen_nodes[addr] = expr_node
    push!(ir.all_nodes, value_node)
    nothing
end

function add_addr!(ir::BasicBlockIR, addr::Symbol, gen::Generator{T,U}, args::Vector, change_expr) where {T,U}
    @assert !ir.finished
    types = get_static_argument_types(gen)
    input_nodes = ValueNode[
        _get_input_node!(ir, expr, typ) for (expr, typ) in zip(args, types)]
    if any([node in ir.incremental_nodes for node in input_nodes])
        incremental_dependency_error(addr)
    end
    change_node = _get_input_node!(ir, change_expr, Expr(:curly, :Nullable, get_change_type(gen)))
    push!(ir.generator_input_change_nodes, change_node)
    expr_node = AddrGeneratorNode(input_nodes, Nullable{ValueNode}(), gen, addr, change_node)
    push!(ir.all_nodes, expr_node)
    ir.addr_gen_nodes[addr] = expr_node
    nothing
end

function add_julia!(ir::BasicBlockIR, expr, typ, name::Symbol)
    @assert !ir.finished
    value_node = ExprValueNode(name, typ)
    ir.value_nodes[name] = value_node
    push!(ir.all_nodes, value_node)
    input_nodes = map((sym) -> ir.value_nodes[sym],
                      collect(resolve_symbols_set(ir, expr)))
    expr_node = JuliaNode(input_nodes, value_node, expr)
    finish!(value_node, expr_node)
    push!(ir.all_nodes, expr_node)
    if any([node in ir.incremental_nodes for node in input_nodes])
        push!(ir.incremental_nodes, value_node)
    end
    value_node
end

function add_argschange!(ir::BasicBlockIR, typ, name::Symbol)
    @assert !ir.finished
    if !isnull(ir.args_change_node)
        error("@argschange can only be called once")
    end
    value_node = ExprValueNode(name, typ)
    ir.value_nodes[name] = value_node
    push!(ir.all_nodes, value_node)
    expr_node = ArgsChangeNode(value_node)
    finish!(value_node, expr_node)
    push!(ir.all_nodes, expr_node)
    push!(ir.incremental_nodes, value_node)
    ir.args_change_node = Nullable(expr_node)
    nothing
end

function add_change!(ir::BasicBlockIR, addr::Symbol, typ, name::Symbol)
    @assert !ir.finished
    value_node = ExprValueNode(name, typ)
    ir.value_nodes[name] = value_node
    push!(ir.all_nodes, value_node)
    if haskey(ir.addr_dist_nodes, addr)
        addr_node = ir.addr_dist_nodes[addr]
    elseif haskey(ir.addr_gen_nodes, addr)
        addr_node = ir.addr_gen_nodes[addr]
    else
        error("@change must occur after an @addr. Address: $addr ")
    end
    expr_node = AddrChangeNode(addr, addr_node, value_node)
    finish!(value_node, expr_node)
    push!(ir.all_nodes, expr_node)
    push!(ir.incremental_nodes, value_node)
    push!(ir.addr_change_nodes, expr_node)
    nothing
end

function set_return!(ir::BasicBlockIR, name::Symbol)
    if !isnull(ir.output_node)
        error("Basic block can only have one return statement, found a second: $name")
    end
    value_node = ir.value_nodes[name]
    if value_node in ir.incremental_nodes
        error("Return value cannot depend on @change or @argschange statements")
    end
    ir.output_node = Nullable(value_node)
end

function set_return!(ir::BasicBlockIR, expr::Expr)
    if !isnull(ir.output_node)
        error("Basic block can only have one return statement, found a second: $name")
    end
    if expr.head != :(::)
        error("Explicit type assert required for return expressions")
    end
    return_expr = expr.args[1]
    typ = expr.args[2]
    ir.output_node = Nullable(_get_input_node!(ir, return_expr, typ))
end

function set_retchange!(ir::BasicBlockIR, name::Symbol)
    if !isnull(ir.retchange_node)
        error("Basic block can only have one @retchange statement, found a second: $name")
    end
    value_node = ir.value_nodes[name]
    ir.retchange_node = Nullable(value_node)
end

function set_retchange!(ir::BasicBlockIR, expr::Expr)
    if !isnull(ir.retchange_node)
        error("Basic block can only have one @retchange statement, found a second: $name")
    end
    if expr.head != :(::)
        error("Explicit type assert required for @retchange expressions")
    end
    retchange_expr = expr.args[1]
    typ = expr.args[2]
    ir.retchange_node = Nullable(_get_input_node!(ir, retchange_expr, typ))
end
