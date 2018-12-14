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

function parse_arg_expr(arg_expr::Symbol)
    arg_name = arg_expr
    typ = Any
    (arg_name, typ, false)
end

function parse_arg_expr(arg_expr::Expr)
    if arg_expr.head == :(::)
        @assert length(arg_expr.args) <= 2
        if length(arg_expr.args) == 1
            arg_name = nothing
            typ = Main.eval(arg_expr.args[1]) # undesirable, but the IR takes the Type value
        elseif length(arg_expr.args) == 2
            if !isa(arg_expr.args[1], Symbol)
                static_dsl_syntax_error(arg_expr.args[1])
            end
            arg_name = arg_expr.args[1]
            typ = Main.eval(arg_expr.args[2]) # undesirable, but the IR takes the Type value
        end
        (arg_name, typ, false)
    elseif arg_expr.head == :macrocall && arg_expr.args[1] == STATIC_DSL_GRAD
        @assert isa(arg_expr.args[2], LineNumberNode)
        (arg_name, typ, _) = parse_arg_expr(arg_expr.args[3])
        (arg_name, typ, true)
    else
        static_dsl_syntax_error(arg_expr)
    end
end

function parse_static_dsl_function_signature!(bindings, builder, sig)
    if (!isa(sig, Expr) || sig.head != :call
        || length(sig.args) < 1 || !isa(sig.args[1], Symbol))
        static_dsl_syntax_error(sig)
    end
    name = sig.args[1]
    for arg_expr in sig.args[2:end]
        (arg_name, typ, compute_grad) = parse_arg_expr(arg_expr)
        if arg_name === nothing
            arg_name = gensym()
        end
        if typ === nothing
            typ = Any
        end
        node = add_argument_node!(builder, name=arg_name, typ=typ,
                                  compute_grad=compute_grad)
        bindings[arg_name] = node
    end
    name
end

function parse_lhs(lhs)
    if isa(lhs, Symbol)
        name = lhs
        typ = Any
    elseif isa(lhs, Expr) && lhs.head == :(::) && isa(lhs.args[1], Symbol)
        name = lhs.args[1]
        typ = Main.eval(lhs.args[2]) # undesirable, but the IR takes the type value
    else
        static_dsl_syntax_error(lhs)
    end
    (name, typ)
end

function resolve_symbols(bindings::Dict{Symbol,StaticIRNode}, symbol::Symbol)
    resolved = Dict{Symbol,StaticIRNode}()
    if haskey(bindings, symbol)
        resolved[symbol] = bindings[symbol]
    end
    resolved
end

function resolve_symbols(bindings::Dict{Symbol,StaticIRNode}, expr::Expr)
    resolved = Dict{Symbol,StaticIRNode}()
    if expr.head == :(.)
        merge!(resolved, resolve_symbols(bindings, expr.args[1]))
    else
        for arg in expr.args
            merge!(resolved, resolve_symbols(bindings, arg))
        end
    end
    resolved
end

function resolve_symbols(bindings::Dict{Symbol,StaticIRNode}, value)
    Dict{Symbol,StaticIRNode}()
end

function parse_julia_expr!(bindings, builder, name::Symbol, typ::Type,
                           expr::Union{Expr,QuoteNode}, diff::Bool)
    resolved = resolve_symbols(bindings, expr)
    inputs = collect(resolved)
    input_symbols = map((x) -> x[1], inputs)
    input_nodes = map((x) -> x[2], inputs)
    fn = Main.eval(Expr(:function, Expr(:tuple, input_symbols...), expr))
    if diff
        add_diff_julia_node!(builder, fn, inputs=input_nodes, name=name, typ=typ)
    else
        # check that none of the inputs are diff nodes
        for node in input_nodes
            if isa(node, DiffNode) 
                error("non-diff expression $expr depends on diff variable $(node.name)")
            end
        end
        add_julia_node!(builder, fn, inputs=input_nodes, name=name, typ=typ)
    end
end

# TODO use add_constant_node! (?)
function parse_julia_expr!(bindings, builder, name::Symbol, typ::Type, value, diff::Bool)
    fn = Main.eval(Expr(:function, Expr(:tuple), QuoteNode(value)))
    if diff
        add_diff_julia_node!(builder, fn, inputs=input_nodes, name=name, typ=typ)
    else
        add_julia_node!(builder, fn, inputs=[], name=name, typ=typ)
    end
end

function parse_julia_expr!(bindings, builder, name::Symbol, typ::Type, expr::Symbol, diff::Bool)
    if haskey(bindings, expr)
        # don't create a new Julia node, just use the existing node
        # NOTE: we aren't using 'typ'
        return bindings[expr]
    end
    parse_julia_expr!(bindings, builder, name, typ, Expr(:block, expr), diff)
end

function parse_julia_assignment!(bindings, builder, line::Expr)
    if line.head != :(=)
        return false
    end
    @assert length(line.args) == 2
    (lhs, expr) = line.args
    (name::Symbol, typ::Type) = parse_lhs(lhs)
    node = parse_julia_expr!(bindings, builder, name, typ, expr, false)
    bindings[name] = node
    true
end

function parse_random_choice_helper!(bindings, builder, name, typ, addr_expr)
    if !(isa(addr_expr, Expr) && addr_expr.head == :macrocall
        && length(addr_expr.args) == 4 && addr_expr.args[1] == STATIC_DSL_ADDR)
        return false
    end
    @assert isa(addr_expr.args[2], LineNumberNode)
    call = addr_expr.args[3]
    if !isa(call, Expr) || call.head != :call
        return false
    end
    local addr::Symbol
    dist = Main.eval(call.args[1])
    if !isa(dist, Distribution)
        return false
    end

    is_choice_node = isa(addr_expr.args[4], QuoteNode) && isa(addr_expr.args[4].value, Symbol)
    if is_choice_node

        # no choice_at or call_at combinator
        addr = addr_expr.args[4].value
        args = call.args[2:end]
        
    else

        # use choice_at combinator(s)
        keys = []
        split_addr!(keys, addr_expr.args[4]) # TODO better error msg
        if !isa(keys[1], QuoteNode) || !isa(keys[1].value, Symbol)
            return false
        end
        @assert length(keys) > 1
        addr = keys[1].value
        gen_fn = choice_at(dist, Any) # TODO use a type annotation
        for i in keys[3:end]
            # TODO do something better than Any -- i.e. use a type annotation
            gen_fn = call_at(gen_fn, Any) 
        end
        args = (call.args[2:end]..., reverse(keys[2:end])...)
    end

    # add input nodes
    inputs = []
    for arg_expr in args
        push!(inputs, parse_julia_expr!(bindings, builder, gensym(), Any, arg_expr, false))
    end

    if is_choice_node

        # random choice node
        node = add_random_choice_node!(builder, dist, inputs=inputs,
                                       addr=addr, name=name, typ=typ)
    else

        # call function node
        argdiff = add_constant_diff_node!(builder, unknownargdiff)
        node = add_gen_fn_call_node!(builder, gen_fn, inputs=inputs,
                                     addr=addr, argdiff=argdiff, name=name, typ=typ)
    end
    bindings[name] = node
    true
end

function parse_random_choice!(bindings, builder, line::Expr)
    if line.head == :(=)
        @assert length(line.args) == 2
        (lhs, rhs) = line.args
        (name::Symbol, typ::Type) = parse_lhs(lhs)
        return parse_random_choice_helper!(bindings, builder, name, typ, rhs)
    else
        name = gensym()
        # TODO this will be inefficient, since it becomes a field in the trace.
        # a concrete type is better.
        # allow a type assertion e.g. @addr(normal(mu, std), :x)::Float64
        typ = Any 
        return parse_random_choice_helper!(bindings, builder, name, typ, line)
    end
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

function parse_gen_fn_call_helper!(bindings, builder, name, typ, addr_expr)
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
    gen_fn = Main.eval(call.args[1])
    if !isa(gen_fn, GenerativeFunction)
        return false
    end

    if isa(addr_expr.args[4], QuoteNode) && isa(addr_expr.args[4].value, Symbol)

        # no call_at combinator
        addr = addr_expr.args[4].value
        args = call.args[2:end]

    else

        # use call_at combinator(s)
        keys = []
        split_addr!(keys, addr_expr.args[4]) # TODO better error msg
        if !isa(keys[1], QuoteNode) || !isa(keys[1].value, Symbol)
            return false
        end
        @assert length(keys) > 1
        addr = keys[1].value
        for key in keys[2:end]
            # TODO do something better than Any -- i.e. use a type annotation
            gen_fn = call_at(gen_fn, Any) 
        end
        # args is a tuple of expressions
        # :(:z) -- it needs to unquote it.. QuoteNode(:z)
        # :z -- it needs to resolve z in the bindings :z
        args = (call.args[2:end]..., reverse(keys[2:end])...)
    end
    
    # add input nodes
    inputs = []
    for arg_expr in args
        push!(inputs, parse_julia_expr!(bindings, builder, gensym(), Any, arg_expr, false))
    end

    # argdiff
    if length(addr_expr.args) == 5
        argdiff_expr = addr_expr.args[5]
        argdiff = parse_julia_expr!(bindings, builder, gensym(), Any, argdiff_expr, true)
    else
        argdiff = add_constant_diff_node!(builder, unknownargdiff)
    end

    # call function node
    node = add_gen_fn_call_node!(builder, gen_fn, inputs=inputs,
                                 addr=addr, argdiff=argdiff, name=name, typ=typ)
    bindings[name] = node
    true
end


function parse_gen_fn_call!(bindings, builder, line::Expr)
    if line.head == :(=)
        @assert length(line.args) == 2
        (lhs, rhs) = line.args
        (name::Symbol, typ::Type) = parse_lhs(lhs)
        return parse_gen_fn_call_helper!(bindings, builder, name, typ, rhs)
    else
        name = gensym()
        typ = Any 
        return parse_gen_fn_call_helper!(bindings, builder, name, typ, line)
    end
    true
end

# return foo (must be a symbol)
function parse_return!(bindings, builder, line::Expr)
    if line.head != :return || !isa(line.args[1], Symbol)
        return false
    end
    var = line.args[1]
    if haskey(bindings, var)
        node = bindings[var]
        set_return_node!(builder, node)
        return true
    else
        error("Tried to return $var, which is not a locally bound variable")
    end
end

# @diff something::typ = @argdiff()
function parse_received_argdiff!(bindings, builder, line::Expr)
    if line.head != :macrocall || length(line.args) != 3 && line.args[1] != STATIC_DSL_DIFF
        return false
    end
    expr = line.args[3]
    if !isa(expr, Expr) || expr.head != :(=) || length(expr.args) != 2
        return false
    end
    lhs = expr.args[1]
    rhs = expr.args[2]
    (name::Symbol, typ::Type) = parse_lhs(lhs)
    if !isa(rhs, Expr) || rhs.head != :macrocall || rhs.args[1] != STATIC_DSL_ARGDIFF
        return false
    end
    node = add_received_argdiff_node!(builder, name=name, typ=typ)
    bindings[name] = node
    true
end

# @diff @retdiff(something)
function parse_retdiff!(bindings, builder, line::Expr)
    if line.head != :macrocall || length(line.args) != 3 && line.args[1] != STATIC_DSL_DIFF
        return false
    end
    expr = line.args[3]
    if (!isa(expr, Expr) || expr.head != :macrocall || length(expr.args) != 3 ||
        expr.args[1] != STATIC_DSL_RETDIFF)
        return false
    end
    inner_expr = expr.args[3]
    node = parse_julia_expr!(bindings, builder, gensym(), Any, inner_expr, true)
    set_retdiff_node!(builder, node)
    true
end

# @diff something::typ = <rhs>
function parse_diff_julia!(bindings, builder, line::Expr)
    if line.head != :macrocall || length(line.args) != 3 && line.args[1] != STATIC_DSL_DIFF
        return false
    end
    expr = line.args[3]
    if !isa(expr, Expr) || expr.head != :(=) || length(expr.args) != 2
        return false
    end
    lhs = expr.args[1]
    rhs = expr.args[2]
    (name::Symbol, typ::Type) = parse_lhs(lhs)
    node = parse_julia_expr!(bindings, builder, name, typ, rhs, true)
    bindings[name] = node
    true 
end

# @diff something[::typ] = @choicediff(:foo)
function parse_choicediff!(bindings, builder, line::Expr)
    if line.head != :macrocall || length(line.args) != 3 && line.args[1] != STATIC_DSL_DIFF
        return false
    end
    expr = line.args[3]
    if !isa(expr, Expr) || expr.head != :(=) || length(expr.args) != 2
        return false
    end
    lhs = expr.args[1]
    rhs = expr.args[2]
    (name::Symbol, typ::Type) = parse_lhs(lhs)
    if (!isa(rhs, Expr) || rhs.head != :macrocall || length(rhs.args) != 3 ||
        rhs.args[1] != STATIC_DSL_CHOICEDIFF ||
        !isa(rhs.args[3], QuoteNode) || !isa(rhs.args[3].value, Symbol))
        return false
    end
    addr::Symbol = rhs.args[3].value
    node = add_choicediff_node!(builder, addr, name=name, typ=typ)
    bindings[name] = node
    true
end

# @diff something::typ = @calldiff(:foo)
function parse_calldiff!(bindings, builder, line::Expr)
    if line.head != :macrocall || length(line.args) != 3 && line.args[1] != STATIC_DSL_DIFF
        return false
    end
    expr = line.args[3]
    if !isa(expr, Expr) || expr.head != :(=) || length(expr.args) != 2
        return false
    end
    lhs = expr.args[1]
    rhs = expr.args[2]
    (name::Symbol, typ::Type) = parse_lhs(lhs)
    if (!isa(rhs, Expr) || rhs.head != :macrocall || length(rhs.args) != 3 ||
        rhs.args[1] != STATIC_DSL_CALLDIFF ||
        !isa(rhs.args[3], QuoteNode) || !isa(rhs.args[3].value, Symbol))
        return false
    end
    addr::Symbol = rhs.args[3].value
    node = add_calldiff_node!(builder, addr, name=name, typ=typ)
    bindings[name] = node
    true
end

function parse_static_dsl_function_body!(bindings, builder, body)
    # TODO use line number nodes to provide better error messages in generated code
    if !isa(body, Expr) || body.head != :block
        static_dsl_syntax_error(body)
    end
    for line in body.args
        isa(line, LineNumberNode) && continue
        !isa(line, Expr) && static_dsl_syntax_error(line)

        # lhs = @addr(rhs..) or @addr(rhs)
        parse_random_choice!(bindings, builder, line) && continue
        parse_gen_fn_call!(bindings, builder, line) && continue

        # lhs = rhs
        # (only run if parsing as choice and call both fail)
        parse_julia_assignment!(bindings, builder, line) && continue

        # return ..
        parse_return!(bindings, builder, line) && continue

        # @diff ..
        parse_received_argdiff!(bindings, builder, line) && continue
        parse_choicediff!(bindings, builder, line) && continue
        parse_calldiff!(bindings, builder, line) && continue
        parse_diff_julia!(bindings, builder, line) && continue
        parse_retdiff!(bindings, builder, line) && continue

        static_dsl_syntax_error(line)
    end
end

function parse_static_dsl_function(ast::Expr)
    if !isa(ast, Expr) || ast.head != :function || length(ast.args) != 2
        static_dsl_syntax_error(ast)
    end
    sig = ast.args[1]
    body = ast.args[2]
    bindings = Dict{Symbol,StaticIRNode}()
    builder = StaticIRBuilder()
    name = parse_static_dsl_function_signature!(bindings, builder, sig)
    parse_static_dsl_function_body!(bindings, builder, body)
    ir = build_ir(builder)
    (name, ir)
end

macro staticgen(ast)

    # parse the AST
    (name, ir)  = parse_static_dsl_function(ast)

    # return code that defines the trace and generator types
    esc(generate_generative_function(ir, name))
end

export @staticgen
