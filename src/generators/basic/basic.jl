#######################
# parsing AST into IR #
#######################

function is_differentiable(typ::Type)
    typ <: AbstractFloat || typ <: AbstractArray{T} where {T <: AbstractFloat}
end

include("ir.jl")

struct BasicBlockParseError <: Exception
    expr::Any
end

function parse_addr(expr::Expr)
    @assert expr.head == :macrocall && expr.args[1] == Symbol("@addr")
    line::LineNumberNode = expr.args[2]
    rest = expr.args[3:end]
    if length(rest) == 2
        change_expr = nothing
    elseif length(rest) == 3
        change_expr = rest[3]
    else
        throw(BasicBlockParseError(expr))
    end
    call = rest[1]
    if !isa(call, Expr) || call.head != :call
        throw(BasicBlockParseError(call))
    end
    generator_or_dist = Main.eval(call.args[1])
    if isa(generator_or_dist, Distribution) && change_expr != nothing
        error("Cannot pass change values to @addr for distributions")
    end
    args = call.args[2:end]
    addr = rest[2]
    if isa(addr, Expr) && length(addr.args) == 1 && isa(addr.args[1], Symbol)
        addr = addr.args[1]
    elseif isa(addr, QuoteNode)
        addr = addr.value
    end
    (addr::Symbol, generator_or_dist, args, change_expr, line)
end

function parse_change(expr::Expr)   
    @assert expr.head == :macrocall && expr.args[1] == Symbol("@change")
    rest = isa(expr.args[2], LineNumberNode) ? expr.args[3:end] : expr.args[2:end]
    if length(rest) != 1
        throw(BasicBlockParseError(expr))
    end
    addr = rest[1]
    if isa(addr, Symbol)
        pass
    elseif isa(addr, Expr) && length(addr.args) == 1 && isa(addr.args[1], Symbol)
        addr = addr.args[1]
    elseif isa(addr, QuoteNode)
        addr = addr.value
    else
        throw(BasicBlockParseError(addr))
    end
    addr::Symbol
end

function parse_lhs(lhs::Expr)
    if lhs.head == :(::)
        name = lhs.args[1]
        typ = Main.eval(lhs.args[2])
        return (name, typ)
    else
        throw(BasicBlockParseError(lhs))
    end
end

function parse_lhs(lhs::Symbol)
    name = lhs
    (name, Any)
end

function parse_param(expr)
    @assert expr.head == :macrocall && expr.args[1] == Symbol("@param")
    rest = isa(expr.args[2], LineNumberNode) ? expr.args[3:end] : expr.args[2:end]
    if length(rest) != 1
        throw(BasicBlockParseError(rest))
    end
    decl = rest[1]
    if isa(decl, Symbol)
        name = decl
        typ = Any
    elseif isa(decl, Expr) && decl.head == :(::)
        name = decl.args[1]
        typ = Main.eval(decl.args[2])
    else
        throw(BasicBlockParseError(rest[1]))
    end
    (name, typ)
end


function generate_ir(args, body, output_ad, args_ad)
    ir = BasicBlockIR(output_ad, args_ad)
    if !isa(body, Expr) || body.head != :block
        throw(BasicBlockParseError(body))
    end
    for arg in args
        if isa(arg, Symbol)
            name = arg
            typ = Any
        elseif isa(arg, Expr) && arg.head == :(::)
            name = arg.args[1]
            typ = Main.eval(arg.args[2])
        else
            throw(BasicBlockParseError(body))
        end
        add_argument!(ir, name, typ)
    end
    local line::LineNumberNode
    for statement in body.args
        if isa(statement, LineNumberNode)
            line = statement
            continue
        end
        if !isa(statement, Expr)
            throw(BasicBlockParseError(statement))
        end
        if statement.head == :macrocall && statement.args[1] == Symbol("@param")
            (name, typ) = parse_param(statement)
            add_param!(ir, name, typ)
        elseif statement.head == :macrocall && statement.args[1] == Symbol("@addr")
            # an @addr statement without a left-hand-side
            (addr, dist_or_gen, args, change_expr, line) = parse_addr(statement)
            if isa(dist_or_gen, Distribution)
                @assert change_expr == nothing
                add_addr!(ir, addr, line, dist_or_gen, args)
            else
                # change_expr may be nothing, indicating nothing is known
                add_addr!(ir, addr, line, dist_or_gen, args, change_expr)
            end
        elseif statement.head == :(=)
            lhs = statement.args[1]
            rhs = statement.args[2]
            (name, typ) = parse_lhs(lhs)
            if rhs.head == :macrocall && rhs.args[1] == Symbol("@addr")
                (addr, dist_or_gen, args, change_expr, line) = parse_addr(rhs)
                if isa(dist_or_gen, Distribution)
                    @assert change_expr == nothing
                    add_addr!(ir, addr, line, dist_or_gen, args, typ, name)
                else
                    # change_expr may be nothing, indicating nothing is known
                    add_addr!(ir, addr, line, dist_or_gen, args, typ, name, change_expr)
                end
            elseif rhs.head == :macrocall && rhs.args[1] == Symbol("@argschange")
                if (length(rhs.args) == 2 && isa(rhs.args[2], LineNumberNode)) || length(rhs.args) == 1
                    add_argschange!(ir, typ, name)
                else
                    dump(rhs)
                    throw(BasicBlockParseError(statement))
                end
            elseif rhs.head == :macrocall && rhs.args[1] == Symbol("@change")
                addr = parse_change(rhs)
                add_change!(ir, addr, typ, name)
            else
                add_julia!(ir, rhs, typ, name, line)
            end
        elseif statement.head == :return
            if length(statement.args) != 1
                throw(BasicBlockParseError(statement))
            end
            expr = statement.args[1]
            if isa(expr, Symbol)
                set_return!(ir, expr)
            elseif isa(expr, Expr) && expr.head == :(::)
                expr_value = expr.args[1]
                typ = Main.eval(expr.args[2])
                set_return!(ir, expr_value, typ)
            else
                throw(BasicBlockParseError(statement))
            end
        elseif statement.head == :macrocall && statement.args[1] == Symbol("@retchange")
            if length(statement.args) != 2
                throw(BasicBlockParseError(statement))
            end
            expr = statement.args[2]
            if isa(expr, Symbol)
                set_retchange!(ir, expr)
            elseif isa(expr, Expr) && expr.head == :(::)
                expr_value = expr.args[1]
                typ = Main.eval(expr.args[2])
                set_retchange!(ir, expr_value, typ)
            else
                throw(BasicBlockParseError(statement))
            end
        else
            throw(BasicBlockParseError(statement))
        end
    end
    finish!(ir)
    ir
end

function compiled_gen_parse_inner(ast)
    @assert isa(ast, Expr) && ast.head == :macrocall && ast.args[1] == Symbol("@gen")
    ast = isa(ast.args[2], LineNumberNode) ? ast.args[3] : ast.args[2]
    if ast.head != :function
        error("syntax error in @compiled at $(ast) in $(ast.head)")
    end
    if length(ast.args) != 2
        error("syntax error in @compile at $(ast) in $(ast.args)")
    end
    signature = ast.args[1]
    body = ast.args[2]
    if signature.head != :call
        error("syntax error in @compiled at $(ast) in $(signature)")
    end
    name = signature.args[1]
    args = signature.args[2:end]
    has_argument_grads = (map(marked_for_ad, args)...,)
    args = map(strip_marked_for_ad, args)
    (name, args, body, has_argument_grads)
end

function compiled_gen_parse(ast)
    if ast.head == :macrocall && ast.args[1] == Symbol("@gen")
        (name, args, body, args_ad) = compiled_gen_parse_inner(ast)
        return (name, args, body, false, args_ad)
    elseif ast.head == :macrocall && ast.args[1] == Symbol("@ad")
        sub_ast = isa(ast.args[2], LineNumberNode) ? ast.args[3] : ast.args[2]
        if (isa(sub_ast, Expr)
            && sub_ast.head == :macrocall
            && sub_ast.args[1] == Symbol("@gen"))
            (name, args, body, args_ad) = compiled_gen_parse_inner(sub_ast)
            return (name, args, body, true, args_ad)
        end
    end 
    error("syntax error in @compiled, expected @compiled @gen .. or @compiled @ad @gen ..")
end

###########################
# basic block trace types #
###########################

# the trace is a mutable struct with fields:
# - a field for each value node (prefixed with a gensym)
# - a field for each addr statement (either a subtrace nor a value)
# - note: there is redundancy between the value nodes and the distribution addr fields

const is_empty_field = gensym("is_empty")
const call_record_field = gensym("call_record")
const value_node_prefix = gensym("value")

function value_field(name::Symbol)
    Symbol("$(value_node_prefix)_$name")
end

function value_field(node::ValueNode)
    value_field(node.name)
end

function value_trace_ref(trace, node::ValueNode)
    fieldname = value_field(node)
    Expr(:(.), trace, QuoteNode(fieldname))
end

struct BasicBlockAssignment{T} <: Assignment
    trace::T
end

static_has_leaf_node(::BasicBlockAssignment, addr) = false
static_has_internal_node(::BasicBlockAssignment, addr) = false

get_address_schema(::Type{BasicBlockAssignment{T}}) where {T} = get_address_schema(T)
Base.isempty(trie::BasicBlockAssignment) = getproperty(trie.trace, is_empty_field)
get_leaf_node(trie::BasicBlockAssignment, key::Symbol) = static_get_leaf_node(trie, Val(key))
get_internal_node(trie::BasicBlockAssignment, key::Symbol) = static_get_internal_node(trie, Val(key))
has_leaf_node(trie::BasicBlockAssignment, key::Symbol) = static_has_leaf_node(trie, Val(key))
has_internal_node(trie::BasicBlockAssignment, key::Symbol) = static_has_internal_node(trie, Val(key))
has_leaf_node(trie::BasicBlockAssignment, addr::Pair) = _has_leaf_node(trie, addr)
get_leaf_node(trie::BasicBlockAssignment, addr::Pair) = _get_leaf_node(trie, addr)
has_internal_node(trie::BasicBlockAssignment, addr::Pair) = _has_internal_node(trie, addr)
get_internal_node(trie::BasicBlockAssignment, addr::Pair) = _get_internal_node(trie, addr)

function make_assignment_methods(trace_type, addr_dist_nodes, addr_gen_nodes)
    methods = Expr[]

    ## get_leaf_nodes ##

    leaf_addrs = map((node) -> node.address, addr_dist_nodes)
    push!(methods, quote
        function Gen.get_leaf_nodes(trie::Gen.BasicBlockAssignment{$trace_type})
            $(Expr(:tuple,
                [quote ($(QuoteNode(addr)), trie.trace.$addr) end for addr in leaf_addrs]...))
        end
    end)

    ## get_internal_nodes ##

    internal_addrs = map((node) -> node.address, addr_gen_nodes)
    push!(methods, quote
        function Gen.get_internal_nodes(trie::Gen.BasicBlockAssignment{$trace_type})
            $(Expr(:tuple,
                [quote ($(QuoteNode(addr)), get_assignment(trie.trace.$addr)) end for addr in internal_addrs]...))
        end
    end)

    for node::AddrDistNode in addr_dist_nodes
        addr = node.address

        ## static_get_leaf_node ##

        push!(methods, quote
            function Gen.static_get_leaf_node(trie::Gen.BasicBlockAssignment{$trace_type},
                                           ::Val{$(QuoteNode(addr))})
                trie.trace.$addr
            end
        end)

        ## static_has_leaf_node ##

        push!(methods, quote
            function Gen.static_has_leaf_node(trie::Gen.BasicBlockAssignment{$trace_type},
                                           ::Val{$(QuoteNode(addr))})
                true
            end
        end)

    end

    for node::AddrGeneratorNode in addr_gen_nodes
        addr = node.address

        ## static_has_internal_node ##

        push!(methods, quote
            function Gen.static_has_internal_node(trie::Gen.BasicBlockAssignment{$trace_type},
                                                  ::Val{$(QuoteNode(addr))})
                true
            end
        end)

        ## static_get_internal_node ##

        push!(methods, quote
            function Gen.static_get_internal_node(trie::Gen.BasicBlockAssignment{$trace_type},
                                                  ::Val{$(QuoteNode(addr))})
                get_assignment(trie.trace.$addr)
            end
        end)
    end

    methods
end

function generate_trace_type(ir::BasicBlockIR, name)
    trace_type_name = gensym("BasicBlockTrace_$name")
    fields = Expr[]
    for (name, node) in ir.value_nodes
        # NOTE: for now record the incremental computations
        # (ir.incremental_nodes) in the trace, but these can be removed for
        # performance optimization
        typ = get_type(node)
        push!(fields, Expr(:(::), value_field(node), QuoteNode(typ)))
    end
    for (addr, node) in ir.addr_dist_nodes
        typ::Type = get_return_type(node.dist)
        push!(fields, Expr(:(::), node.address, QuoteNode(typ)))
    end
    for (addr, node) in ir.addr_gen_nodes
        typ::Type = get_trace_type(node.gen)
        push!(fields, Expr(:(::), node.address, QuoteNode(typ)))
    end
    addresses = union(keys(ir.addr_dist_nodes), keys(ir.addr_gen_nodes))
    assignment_methods = make_assignment_methods(
        trace_type_name, values(ir.addr_dist_nodes), values(ir.addr_gen_nodes))
    retval_type = QuoteNode(ir.output_node ===  nothing ? Nothing : ir.output_node.typ)
    defn = esc(quote

        # specialized trace implementation
        mutable struct $trace_type_name
            $is_empty_field::Bool
            $call_record_field::CallRecord{$retval_type}
            $(Expr(:block, fields...))
            $trace_type_name() = new()
        end

        function Base.copy(other::$trace_type_name)
            trace = $trace_type_name()
            $(Expr(:block, [
                let fieldname = field.args[1] 
                    quote trace.$fieldname = other.$fieldname end
                end for field in fields]...))
            trace
        end

        Gen.get_call_record(trace::$trace_type_name) = trace.$call_record_field
        Gen.has_choices(trace::$trace_type_name) = !trace.$is_empty_field

        # assignment view of the trace
        Gen.get_assignment(trace::$trace_type_name) = Gen.BasicBlockAssignment(trace)
        
        function Gen.get_address_schema(::Type{$trace_type_name})
            Gen.StaticAddressSchema(
                Set{Symbol}([$([QuoteNode(addr) for addr in keys(ir.addr_dist_nodes)]...)]),
                Set{Symbol}([$([QuoteNode(addr) for addr in keys(ir.addr_gen_nodes)]...)]))
        end
        $(Expr(:block, assignment_methods...))
    end)
    (defn, trace_type_name)
end


#########################
# basic block generator #
#########################

abstract type BasicGenFunction{T,U} <: Generator{T,U} end

# a method on the generator type that is executed during expansion of
# generator API generated functions
function get_ir end
function get_grad_fn end

function generate_generator_type(ir::BasicBlockIR, trace_type::Symbol, name::Symbol, node_to_gradient)
    generator_type = gensym("BasicBlockGenerator_$name")
    retval_type = ir.output_node === nothing ? :Nothing : ir.output_node.typ
    defn = esc(quote
        struct $generator_type <: Gen.BasicGenFunction{$retval_type, $trace_type}
            params_grad::Dict{Symbol,Any}
            params::Dict{Symbol,Any}
        end
        $generator_type() = $generator_type(Dict{Symbol,Any}(), Dict{Symbol,Any}())

        (gen::$generator_type)(args...) = get_call_record(simulate(gen, args)).retval
        Gen.get_ir(::Type{$generator_type}) = $(QuoteNode(ir))
        #Gen.render_graph(::$generator_type, fname) = Gen.render_graph(Gen.get_ir($generator_type), fname)
        Gen.get_trace_type(::Type{$generator_type}) = $trace_type
        function Gen.get_static_argument_types(::$generator_type)
            [node.typ for node in Gen.get_ir($generator_type).arg_nodes]
        end
        Gen.accepts_output_grad(::$generator_type) = $(QuoteNode(ir.output_ad))
        Gen.has_argument_grads(::$generator_type) = $(QuoteNode(ir.args_ad))
        Gen.get_grad_fn(::Type{$generator_type}, node::Gen.JuliaNode) = $(QuoteNode(node_to_gradient))[node]
    end)
    (defn, generator_type)
end

# TODO refactor and simplify:
function generate_gradient_fn(node::JuliaNode, gradient_fn::Symbol)
    if isa(node.expr_or_value, Expr) || isa(node.expr_or_value, Symbol)
        input_nodes = node.input_nodes
        inputs_do_ad = map((in_node) -> is_differentiable(get_type(in_node)), node.input_nodes)
        untracked_inputs = [gensym("untracked_$(in_node.name)") for in_node in node.input_nodes]
        maybe_tracked_inputs = [in_node.name for in_node in node.input_nodes]
        track_stmts = Expr[]
        grad_exprs = Expr[]
        grad_exprs_noop = Expr[]
        tape = gensym("tape")
        for (untracked, maybe_tracked, do_ad) in zip(untracked_inputs, maybe_tracked_inputs, inputs_do_ad)
            if do_ad
                push!(track_stmts, quote $maybe_tracked = ReverseDiff.track($untracked, $tape) end)
                push!(grad_exprs, quote ReverseDiff.deriv($maybe_tracked) end)
                push!(grad_exprs_noop, quote zero($untracked) end)
            else
                push!(track_stmts, quote $maybe_tracked = $untracked end)
                push!(grad_exprs, quote nothing end)
                push!(grad_exprs_noop, quote nothing end)
            end
        end
        output_grad = gensym("output_grad")
        given_output_value = gensym("given_output_value")
        output_value_maybe_tracked = gensym("output_value_maybe_tracked")
        err_msg = QuoteNode("julia expression was not differentiable: $(node.expr_or_value)")
        quote
            function $gradient_fn($output_grad, $given_output_value, $(untracked_inputs...))
                $tape = ReverseDiff.InstructionTape()
                $(track_stmts...)
                $output_value_maybe_tracked = $(node.expr_or_value)
                @assert isapprox(ReverseDiff.value($output_value_maybe_tracked), $given_output_value)
                if $output_grad !== nothing
                    if ReverseDiff.istracked($output_value_maybe_tracked)
                        ReverseDiff.deriv!($output_value_maybe_tracked, $output_grad)
                        ReverseDiff.reverse_pass!($tape)
                        return ($(grad_exprs...),)
                    else
                        # the output value was not tracked

                        # this could indicate that the expresssion was not
                        # differentiable, which should probably be an error

                        # or, it could indicate that the expression was a constant

                        # TODO revisit

                        return ($(grad_exprs_noop...),)
                    end
                else
                    # output_grad is nothing (i.e. not a floating point value)
                    return ($(grad_exprs_noop...),)
                end
            end
        end
    else
        # it is a constant value
        @assert length(node.input_nodes) == 0
        quote
            $gradient_fn(output_grad, given_output_value) = ()
        end
    end
end

function generate_gradient_functions(ir::BasicBlockIR)
    gradient_function_defns = Expr[]
    node_to_gradient = Dict{JuliaNode,Symbol}()
    for node::JuliaNode in filter((node) -> isa(node, JuliaNode), ir.all_nodes)
        gradient_fn = gensym("julia_grad_$(node.output.name)")
        gradient_fn_defn = esc(generate_gradient_fn(node, gradient_fn))
        push!(gradient_function_defns, gradient_fn_defn)
        node_to_gradient[node] = gradient_fn
    end
    (gradient_function_defns, node_to_gradient)
end

macro compiled(ast)

    # parse the AST
    (name, args, body, output_ad, args_ad) = compiled_gen_parse(ast)

    # geneate intermediate data-flow representation
    ir = generate_ir(args, body, output_ad, args_ad)

    # generate gradient functions
    (gradient_function_defns, node_to_gradient) = generate_gradient_functions(ir)

    # generate trace type definition
    (trace_type_defn, trace_type) = generate_trace_type(ir, name)

    # generate generator type definition
    (generator_type_defn, generator_type) = generate_generator_type(
        ir, trace_type, name, node_to_gradient)

    Expr(:block,
        trace_type_defn,
        generator_type_defn,
        gradient_function_defns...,
        quote $(esc(name)) = $(esc(generator_type))() end)
end


#####################
# static parameters #
#####################

# V1: just use a dictionary
# V2: create specialized fields.

# note that parameters will be cached (as specialized fields) in the trace;
# user will need to use assess() after changing the parameters to get a trace
# that has the new values of the parameters, before doing e.g. backprop()

# for V1, just during simulate and assess, the parameters will be read from
# dictionaries

function set_param!(gf::BasicGenFunction, name::Symbol, value)
    gf.params[name] = value
end

function get_param(gf::BasicGenFunction, name::Symbol)
    gf.params[name]
end

function get_param_grad(gf::BasicGenFunction, name::Symbol)
    gf.params_grad[name]
end

function zero_param_grad!(gf::BasicGenFunction, name::Symbol)
    gf.params_grad[name] = zero(gf.params[name])
end

function init_param!(gf::BasicGenFunction, name::Symbol, value)
    set_param!(gf, name, value)
    zero_param_grad!(gf, name)
end


######################
# change propagation #
######################

"""
Example: MaskedArgChange{Tuple{Val{:true},Val{:false}},Something}(something)
"""
struct MaskedArgChange{T <: Tuple,U}
    info::U
end

# TODO make the type parameter U part of the BasicGenFunction type parameter?
get_change_type(::BasicGenFunction) = MaskedArgChange

function mask(bits...)
    parameters = map((bit) -> Val{bit}, bits)
    MaskedArgChange{Tuple{parameters...},Nothing}(nothing)
end

export MaskedArgChange, mask

include("simulate.jl")
include("assess.jl")
include("generate.jl")
include("update.jl")
include("backprop.jl")

export @compiled
