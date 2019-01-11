const STATIC_DSL_GRAD = Symbol("@grad")
const STATIC_DSL_ADDR = Symbol("@addr")
const STATIC_DSL_DIFF = Symbol("@diff")
const STATIC_DSL_CHOICEDIFF = Symbol("@choicediff")
const STATIC_DSL_CALLDIFF = Symbol("@calldiff")
const STATIC_DSL_ARGDIFF = Symbol("@argdiff")
const STATIC_DSL_RETDIFF = Symbol("@retdiff")

function static_dsl_syntax_error(expr)
    error("Syntax error when parsing static DSL function at $expr")
end

function parse_lhs(lhs)
    if isa(lhs, Symbol)
        name = lhs
        typ = QuoteNode(Any)
    elseif isa(lhs, Expr) && lhs.head == :(::) && isa(lhs.args[1], Symbol)
        name = lhs.args[1]
        typ = lhs.args[2]
    else
        static_dsl_syntax_error(lhs)
    end
    (name, typ)
end

function resolve_symbols(bindings::Dict{Symbol,Symbol}, symbol::Symbol)
    resolved = Dict{Symbol,Symbol}()
    if haskey(bindings, symbol)
        resolved[symbol] = bindings[symbol]
    end
    resolved
end

function resolve_symbols(bindings::Dict{Symbol,Symbol}, expr::Expr)
    resolved = Dict{Symbol,Symbol}()
    if expr.head == :(.)
        merge!(resolved, resolve_symbols(bindings, expr.args[1]))
    else
        for arg in expr.args
            merge!(resolved, resolve_symbols(bindings, arg))
        end
    end
    resolved
end

function resolve_symbols(bindings::Dict{Symbol,Symbol}, value)
    Dict{Symbol,Symbol}()
end

# the IR builder needs to contain a bindings map from symbol to IRNode, to
# provide us with input_nodes.

# the macro expansion also needs a bindings set of symbols to resolve from, so that
# we can then insert the loo

function parse_julia_expr!(stmts, bindings, name::Symbol, typ,
                           expr::Union{Expr,QuoteNode}, diff::Bool)
    resolved = resolve_symbols(bindings, expr)
    inputs = collect(resolved)
    input_vars = map((x) -> esc(x[1]), inputs)
    input_nodes = map((x) -> esc(x[2]), inputs)
    fn = Expr(:function, Expr(:tuple, input_vars...), esc(expr))
    node = gensym()
    method = diff ? :add_diff_julia_node! : :add_julia_node!
    push!(stmts, :($(esc(node)) = $(esc(method))(
        builder, $fn, inputs=[$(input_nodes...)], name=$(QuoteNode(name)), typ=$(esc(typ)))))
    node
end

function parse_julia_expr!(stmts, bindings, name::Symbol, typ, value, diff::Bool)
    fn = Expr(:function, Expr(:tuple), QuoteNode(value))
    node = gensym()
    method = diff ? :add_diff_julia_node! : :add_julia_node!
    push!(stmts, :($(esc(node)) = $(esc(method))(
        builder, $fn, inputs=[], name=$(QuoteNode(name)), typ=$(esc(typ)))))
    node
end

function parse_julia_expr!(stmts, bindings, name::Symbol, typ, var::Symbol, diff::Bool)
    if haskey(bindings, var)
        # don't create a new Julia node, just use the existing node
        # NOTE: we aren't using 'typ'
        return bindings[var]
    end
    parse_julia_expr!(stmts, bindings, name, typ, Expr(:block, esc(var)), diff)
end

function parse_julia_assignment!(stmts, bindings, line::Expr)
    if line.head != :(=)
        return false
    end
    @assert length(line.args) == 2
    (lhs, expr) = line.args
    (name::Symbol, typ) = parse_lhs(lhs)
    node = parse_julia_expr!(stmts, bindings, name, typ, expr, false)
    bindings[name] = node
    true
end

split_addr!(keys, addr_expr::QuoteNode) = push!(keys, addr_expr)
split_addr!(keys, addr_expr::Symbol) = push!(keys, addr_expr)

function split_addr!(keys, addr_expr::Expr)
    @assert addr_expr.head == :call
    @assert length(addr_expr.args) == 3
    @assert addr_expr.args[1] == :(=>)
    push!(keys, addr_expr.args[2])
    split_addr!(keys, addr_expr.args[3])
end

choice_or_call_at(gen_fn::GenerativeFunction, addr) = call_at(gen_fn, addr)
choice_or_call_at(dist::Distribution, addr) = choice_at(dist, addr)

function parse_addr_expr!(stmts, bindings, name, typ, addr_expr)
    if !(isa(addr_expr, Expr) && addr_expr.head == :macrocall
        && length(addr_expr.args) >= 4 && addr_expr.args[1] == STATIC_DSL_ADDR)
        return false
    end
    @assert isa(addr_expr.args[2], LineNumberNode)
    call = addr_expr.args[3]
    if !isa(call, Expr) || call.head != :call
        return false
    end
    local addr::Symbol
    gen_fn_or_dist = gensym()
    push!(stmts, :($(esc(gen_fn_or_dist)) = $(esc(call.args[1]))))
    if isa(addr_expr.args[4], QuoteNode) && isa(addr_expr.args[4].value, Symbol)
        addr = addr_expr.args[4].value
        args = call.args[2:end]
    else
        # multi-part address syntactic sugar
        keys = []
        split_addr!(keys, addr_expr.args[4])
        if !isa(keys[1], QuoteNode) || !isa(keys[1].value, Symbol)
            return false
        end
        @assert length(keys) > 1
        addr = keys[1].value
        for key in keys[2:end]
            push!(stmts, :($(esc(gen_fn_or_dist)) = choice_or_call_at($(esc(gen_fn_or_dist)), $(QuoteNode(Any)))))
        end
        args = (call.args[2:end]..., reverse(keys[2:end])...)
    end
    node = gensym()
    bindings[name] = node
    inputs = []
    for arg_expr in args
        push!(inputs, parse_julia_expr!(stmts, bindings, gensym(), QuoteNode(Any), arg_expr, false))
    end
    if length(addr_expr.args) == 5
        # argdiff
        argdiff_expr = addr_expr.args[5]
        argdiff = parse_julia_expr!(stmts, bindings, gensym(), QuoteNode(Any), argdiff_expr, true)
        push!(stmts, :($(esc(node)) = add_addr_node!(
            builder, $(esc(gen_fn_or_dist)), inputs=[$(map(esc, inputs)...)], addr=$(QuoteNode(addr)),
            argdiff=$(esc(argdiff)), name=$(QuoteNode(name)), typ=$(esc(typ)))))
    else
        push!(stmts, :($(esc(node)) = add_addr_node!(
            builder, $(esc(gen_fn_or_dist)), inputs=[$(map(esc, inputs)...)], addr=$(QuoteNode(addr)),
            name=$(QuoteNode(name)), typ=$(esc(typ)))))
    end
    true
end

function parse_addr_line!(stmts::Vector{Expr}, bindings, line::Expr)
    if line.head == :(=)
        @assert length(line.args) == 2
        (lhs, rhs) = line.args
        (name::Symbol, typ) = parse_lhs(lhs)
        parse_addr_expr!(stmts, bindings, name, typ, rhs)
    else
        name = gensym()
        typ = QuoteNode(Any)
        parse_addr_expr!(stmts, bindings, name, typ, line)
    end
end

# return foo (must be a symbol)
function parse_return!(stmts::Vector{Expr}, bindings, line::Expr)
    if line.head != :return || !isa(line.args[1], Symbol)
        return false
    end
    var = line.args[1]
    if haskey(bindings, var)
        node = bindings[var]
        push!(stmts, :(set_return_node!(builder, $(esc(node)))))
        return true
    else
        error("Tried to return $var, which is not a locally bound variable")
    end
end

# @diff something::typ = @argdiff()
function parse_received_argdiff!(stmts::Vector{Expr}, bindings, line::Expr)
    if line.head != :macrocall || length(line.args) != 3 && line.args[1] != STATIC_DSL_DIFF
        return false
    end
    expr = line.args[3]
    if !isa(expr, Expr) || expr.head != :(=) || length(expr.args) != 2
        return false
    end
    lhs = expr.args[1]
    rhs = expr.args[2]
    (name::Symbol, typ) = parse_lhs(lhs)
    if !isa(rhs, Expr) || rhs.head != :macrocall || rhs.args[1] != STATIC_DSL_ARGDIFF
        return false
    end
    node = gensym()
    push!(stmts, :($(esc(node)) = add_received_argdiff_node!(
        builder, name=$(QuoteNode(name)), typ=$(esc(typ)))))
    bindings[name] = node
    true
end

# @diff @retdiff(something)
function parse_retdiff!(stmts::Vector{Expr}, bindings, line::Expr)
    if line.head != :macrocall || length(line.args) != 3 && line.args[1] != STATIC_DSL_DIFF
        return false
    end
    expr = line.args[3]
    if (!isa(expr, Expr) || expr.head != :macrocall || length(expr.args) != 3 ||
        expr.args[1] != STATIC_DSL_RETDIFF)
        return false
    end
    inner_expr = expr.args[3]
    node = parse_julia_expr!(stmts, bindings, gensym(), QuoteNode(Any), inner_expr, true)
    push!(stmts, :(set_retdiff_node!(builder, $(esc(node)))))
    true
end

# @diff something::typ = <rhs>
function parse_diff_julia!(stmts::Vector{Expr}, bindings, line::Expr)
    if line.head != :macrocall || length(line.args) != 3 && line.args[1] != STATIC_DSL_DIFF
        return false
    end
    expr = line.args[3]
    if !isa(expr, Expr) || expr.head != :(=) || length(expr.args) != 2
        return false
    end
    lhs = expr.args[1]
    rhs = expr.args[2]
    (name::Symbol, typ) = parse_lhs(lhs)
    node = parse_julia_expr!(stmts, bindings, name, typ, rhs, true)
    bindings[name] = node
    true 
end

## @diff something[::typ] = @choicediff(:foo)
function parse_choicediff!(stmts::Vector{Expr}, bindings, line::Expr)
    if line.head != :macrocall || length(line.args) != 3 && line.args[1] != STATIC_DSL_DIFF
        return false
    end
    expr = line.args[3]
    if !isa(expr, Expr) || expr.head != :(=) || length(expr.args) != 2
        return false
    end
    lhs = expr.args[1]
    rhs = expr.args[2]
    (name::Symbol, typ) = parse_lhs(lhs)
    if (!isa(rhs, Expr) || rhs.head != :macrocall || length(rhs.args) != 3 ||
        rhs.args[1] != STATIC_DSL_CHOICEDIFF ||
        !isa(rhs.args[3], QuoteNode) || !isa(rhs.args[3].value, Symbol))
        return false
    end
    addr::Symbol = rhs.args[3].value
    node = gensym()
    push!(stmts, :($(esc(node)) = add_choicediff_node!(
        builder, $(QuoteNode(addr)), name=$(QuoteNode(name)), typ=$(esc(typ)))))
    bindings[name] = node
    true
end

# @diff something::typ = @calldiff(:foo)
function parse_calldiff!(stmts::Vector{Expr}, bindings, line::Expr)
    if line.head != :macrocall || length(line.args) != 3 && line.args[1] != STATIC_DSL_DIFF
        return false
    end
    expr = line.args[3]
    if !isa(expr, Expr) || expr.head != :(=) || length(expr.args) != 2
        return false
    end
    lhs = expr.args[1]
    rhs = expr.args[2]
    (name::Symbol, typ) = parse_lhs(lhs)
    if (!isa(rhs, Expr) || rhs.head != :macrocall || length(rhs.args) != 3 ||
        rhs.args[1] != STATIC_DSL_CALLDIFF ||
        !isa(rhs.args[3], QuoteNode) || !isa(rhs.args[3].value, Symbol))
        return false
    end
    addr::Symbol = rhs.args[3].value
    node = gensym()
    push!(stmts, :($(esc(node)) = add_calldiff_node!(
        builder, $(QuoteNode(addr)), name=$(QuoteNode(name)), typ=$(esc(typ)))))
    bindings[name] = node
    true
end

function parse_static_dsl_function_body!(stmts::Vector{Expr},
                                         bindings::Dict{Symbol,Symbol},
                                         expr)
    # TODO use line number nodes to provide better error messages in generated code
    if !isa(expr, Expr) || expr.head != :block
        static_dsl_syntax_error(expr)
    end
    for line in expr.args
        isa(line, LineNumberNode) && continue
        !isa(line, Expr) && static_dsl_syntax_error(line)

        # lhs = @addr(rhs..) or @addr(rhs)
        parse_addr_line!(stmts, bindings, line) && continue

        # lhs = rhs
        # (only run if parsing as choice and call both fail)
        parse_julia_assignment!(stmts, bindings, line) && continue

        # return ..
        parse_return!(stmts, bindings, line) && continue

        # @diff ..
        parse_received_argdiff!(stmts, bindings, line) && continue
        parse_choicediff!(stmts, bindings, line) && continue
        parse_calldiff!(stmts, bindings, line) && continue
        parse_diff_julia!(stmts, bindings, line) && continue
        parse_retdiff!(stmts, bindings, line) && continue

        static_dsl_syntax_error(line)
    end
end

function make_static_gen_function(name, args, body, return_type, annotations)
    # generate code that builds the IR, then generates code from it and evaluates it
    stmts = Expr[]
    push!(stmts, :(bindings = Dict{Symbol, StaticIRNode}()))
    push!(stmts, :(builder = StaticIRBuilder())) # NOTE: we are relying on the gensym
    bindings = Dict{Symbol,Symbol}() # map from variable name to node name
    for arg in args
        node = gensym()
        push!(stmts, :($(esc(node)) = add_argument_node!(
            builder, name=$(QuoteNode(arg.name)), typ=$(esc(arg.typ)),
            compute_grad=$(QuoteNode(DSL_ARG_GRAD_ANNOTATION in arg.annotations)))))
        bindings[arg.name] = node 
    end
    parse_static_dsl_function_body!(stmts, bindings, body)
    push!(stmts, :(ir = build_ir(builder)))
    push!(stmts, :($(esc(name)) = eval(generate_generative_function(ir, $(QuoteNode(name))))))
    Expr(:block, stmts...)
end
