const STATIC_DSL_GRAD = Symbol("@grad")
const STATIC_DSL_TRACE = Symbol("@trace")
const STATIC_DSL_PARAM = Symbol("@param")

function static_dsl_syntax_error(expr, msg="")
    error("Syntax error when parsing static DSL function at $expr. $msg")
end

"Look-up and return node names bound to each symbol in an expression."
function resolve_symbols(bindings::Dict{Symbol,Symbol}, expr::Expr)
    resolved = Dict{Symbol,Symbol}()
    if expr.head == :(.)
        merge!(resolved, resolve_symbols(bindings, expr.args[1]))
    else
        for arg in expr.args
            merge!(resolved, resolve_symbols(bindings, arg))
        end
    end
    return resolved
end

function resolve_symbols(bindings::Dict{Symbol,Symbol}, symbol::Symbol)
    resolved = Dict{Symbol,Symbol}()
    if haskey(bindings, symbol)
        resolved[symbol] = bindings[symbol]
    end
    return resolved
end

function resolve_symbols(bindings::Dict{Symbol,Symbol}, value)
    Dict{Symbol,Symbol}()
end

"Parse optionally typed variable expressions."
function parse_typed_var(expr)
    if MacroTools.@capture(expr, var_Symbol)
        return (var, QuoteNode(Any))
    elseif MacroTools.@capture(expr, var_Symbol::typ_)
        return (var, typ)
    else
        static_dsl_syntax_error(expr)
    end
end

"Split nested addresses into list of keys."
function split_addr!(keys, addr_expr::Expr)
    @assert MacroTools.@capture(addr_expr, fst_ => snd_)
    push!(keys, fst)
    split_addr!(keys, snd)
end
split_addr!(keys, addr_expr::QuoteNode) = push!(keys, addr_expr)
split_addr!(keys, addr_expr::Symbol) = push!(keys, addr_expr)

"Generate informative node name for a Julia expression."
gen_node_name(arg::Any) = gensym(string(arg))
gen_node_name(arg::Expr) = gensym(arg.head)
gen_node_name(arg::Symbol) = gensym(arg)
gen_node_name(arg::QuoteNode) = gensym(string(arg.value))

"Parse @trace expression and add corresponding node to IR."
function parse_trace_expr!(stmts, bindings, fn, args, addr)
    expr_s = "$STATIC_DSL_TRACE($fn($(join(args, ", "))), $addr)"
    name = gen_node_name(addr) # Each @trace node is named after its address
    node = gen_node_name(addr) # Generate a variable name for the StaticIRNode
    bindings[name] = node
    # Add statement that creates reference to the gen_fn / dist
    gen_fn_or_dist = gensym(string(fn))
    push!(stmts, :($(esc(gen_fn_or_dist)) = $(esc(fn))))
    # Handle the trace address
    keys = []
    split_addr!(keys, addr) # Split nested addresses
    if !(isa(keys[1], QuoteNode) && isa(keys[1].value, Symbol))
        static_dsl_syntax_error(addr, "$(keys[1].value) is not a Symbol")
    end
    addr = keys[1].value # Get top level address
    if length(keys) > 1
        # For each nesting level, wrap gen_fn_or_dist within call_at
        for key in keys[2:end]
            push!(stmts, :($(esc(gen_fn_or_dist)) =
                call_at($(esc(gen_fn_or_dist)), Any)))
        end
        # Append the nested addresses as arguments to call_at
        args = [args; reverse(keys[2:end])]
    end
    # Handle arguments to the traced call
    inputs = []
    for arg_expr in args
        if MacroTools.@capture(arg_expr, x_...)
            static_dsl_syntax_error(expr_s, "Cannot splat in @trace call.")
        end
        # Create Julia node for each argument to gen_fn_or_dist
        arg_name = gen_node_name(arg_expr)
        push!(inputs, parse_julia_expr!(stmts, bindings,
            arg_name, arg_expr, QuoteNode(Any)))
    end
    # Add addr node (a GenerativeFunctionCallNode or RandomChoiceNode)
    push!(stmts, :($(esc(node)) = add_addr_node!(
        builder, $(esc(gen_fn_or_dist)), inputs=[$(map(esc, inputs)...)],
        addr=$(QuoteNode(addr)), name=$(QuoteNode(name)))))
    # Return the name of the newly created node
    return name
end

"Parse a Julia expression and add a corresponding node to the IR."
function parse_julia_expr!(stmts, bindings, name::Symbol, expr::Expr,
                           typ::Union{Symbol,Expr,QuoteNode})
    resolved = resolve_symbols(bindings, expr)
    inputs = collect(resolved)
    input_vars = map((x) -> esc(x[1]), inputs)
    input_nodes = map((x) -> esc(x[2]), inputs)
    fn = Expr(:function, Expr(:tuple, input_vars...), esc(expr))
    node = gensym(name)
    push!(stmts, :($(esc(node)) = add_julia_node!(
        builder, $fn, inputs=[$(input_nodes...)], name=$(QuoteNode(name)),
        typ=$(QuoteNode(typ)))))
    return node
end

function parse_julia_expr!(stmts, bindings, name::Symbol, var::Symbol,
                           typ::Union{Symbol,Expr,QuoteNode})
    if haskey(bindings, var)
        # Use the existing node instead of creating a new one
        return bindings[var]
    end
    node = parse_julia_expr!(stmts, bindings, name, Expr(:block, var), typ)
    return node
end

function parse_julia_expr!(stmts, bindings, name::Symbol, var::QuoteNode,
                           typ::Union{Symbol,Expr,QuoteNode})
    fn = Expr(:function, Expr(:tuple), var)
    node = gensym(name)
    push!(stmts, :($(esc(node)) = add_julia_node!(
        builder, $fn, inputs=[], name=$(QuoteNode(name)),
        typ=$(QuoteNode(typ)))))
    return node
end

function parse_julia_expr!(stmts, bindings, name::Symbol, value,
                           typ::Union{Symbol,Expr,QuoteNode})
    fn = Expr(:function, Expr(:tuple), QuoteNode(value))
    node = gensym(name)
    push!(stmts, :($(esc(node)) = add_julia_node!(
        builder, $fn, inputs=[], name=$(QuoteNode(name)),
        typ=$(QuoteNode(typ)))))
    return node
end

"Parse @param line and add corresponding trainable param node."
function parse_param_line!(stmts::Vector{Expr}, bindings, expr::Expr)
    (name::Symbol, typ) = parse_typed_var(expr)
    if haskey(bindings, name)
        static_dsl_syntax_error(expr, "Symbol $name already bound")
    end
    node = gensym(name)
    bindings[name] = node
    push!(stmts, :($(esc(node)) = add_trainable_param_node!(
        builder, $(QuoteNode(name)), typ=$(QuoteNode(typ)))))
    true
end

"Parse assignments and add corresponding nodes for the right-hand-side."
function parse_assignment_line!(stmts, bindings, lhs, rhs)
    if isa(lhs, Expr) && lhs.head == :tuple
        # Recursively handle tuple assignments
        name, typ = gen_node_name(rhs), QuoteNode(Any)
        node = parse_julia_expr!(stmts, bindings, name, rhs, typ)
        bindings[name] = node
        for (i, lhs_i) in enumerate(lhs.args)
            # Assign lhs[i] = rhs[i]
            rhs_i = :($name[$i])
            parse_assignment_line!(stmts, bindings, lhs_i, rhs_i)
        end
    else
        # Handle single variable assignment (base case)
        (name::Symbol, typ) = parse_typed_var(lhs)
        # Generate new node name if name is already bound
        node_name = haskey(bindings, name) ? gensym(name) : name
        node = parse_julia_expr!(stmts, bindings, node_name, rhs, typ)
        # Old bindings are overwritten with new nodes
        bindings[name] = node
    end
    # Return name of node to be processed by parent expressions
    return name
end

"Parse a return line and add corresponding return node."
function parse_return_line!(stmts, bindings, expr)
    if isa(expr, Symbol)
        if !haskey(bindings, expr)
            error("Tried to return $expr, which is not a locally bound variable")
        end
        node = bindings[expr]
    else
        name, typ = gensym("return"), QuoteNode(Any)
        node = parse_julia_expr!(stmts, bindings, name, expr, typ)
        bindings[name] = node
    end
    push!(stmts, :(set_return_node!(builder, $(esc(node)))))
    return Expr(:return, expr)
end

"Parse and rewrite expression if it matches an @trace call."
function parse_and_rewrite_trace!(stmts, bindings, expr)
    if (MacroTools.@capture(expr, @m_(f_(xs__), addr_)) && isa(m, GlobalRef) &&
        m.name == STATIC_DSL_TRACE && m.mod == @__MODULE__)
        # Parse "@trace(f(xs...), addr)" and return fresh variable
        parse_trace_expr!(stmts, bindings, f, xs, addr)
    elseif (MacroTools.@capture(expr, @m_(f_(xs__))) && isa(m, GlobalRef) &&
            m.name == STATIC_DSL_TRACE && m.mod == @__MODULE__)
        # Throw error for @trace expression without address
        static_dsl_syntax_error(expr, "Address required.")
    else
        expr # Return expression unmodified
    end
end

"Parse line (i.e. top-level expression) of a static Gen function body."
function parse_static_dsl_line!(stmts, bindings, line)
    # Walk each line bottom-up, parsing and rewriting @trace expressions
    rewritten = MacroTools.postwalk(
        e -> parse_and_rewrite_trace!(stmts, bindings, e), line)
    # If line is a top-level @trace call, we are done
    if (MacroTools.@capture(line, @m_(f_(x__), a_)) && isa(m, GlobalRef) &&
        m.name == STATIC_DSL_TRACE && m.mod == @__MODULE__)
        return
    end
    # Match and parse any other top-level expressions
    line = rewritten
    if (MacroTools.@capture(line, @m_(expr_)) && isa(m, GlobalRef) &&
        m.name == STATIC_DSL_PARAM && m.mod == @__MODULE__)
        # Parse "@param var::T"
        parse_param_line!(stmts, bindings, expr)
    elseif MacroTools.@capture(line, lhs_ = rhs_)
        # Parse "lhs = rhs"
        parse_assignment_line!(stmts, bindings, lhs, rhs)
    elseif MacroTools.@capture(line, return expr_)
        # Parse "return expr"
        parse_return_line!(stmts, bindings, expr)
    elseif line isa LineNumberNode
        # Skip line number nodes
    else
        # Disallow all other top-level constructs
        static_dsl_syntax_error(line, "Unsupported top-level construct.")
    end
end

"Parse static Gen function body line by line."
function parse_static_dsl_function_body!(
    stmts::Vector{Expr}, bindings::Dict{Symbol,Symbol}, expr)
    # TODO: Use line number nodes to improve error messages in generated code
    if !(isa(expr, Expr) && expr.head == :block)
        static_dsl_syntax_error(expr)
    end
    for line in expr.args
        parse_static_dsl_line!(stmts, bindings, line)
    end
end

"Generates the code that builds the IR of a static Gen function."
function make_static_gen_function(name, args, body, return_type, annotations)
    # Construct the builder for the intermediate representation (IR)
    stmts = Expr[]
    push!(stmts, :(bindings = Dict{Symbol, StaticIRNode}()))
    push!(stmts, :(builder = StaticIRBuilder()))
    accepts_output_grad = DSL_RET_GRAD_ANNOTATION in annotations
    push!(stmts, :(set_accepts_output_grad!(builder, $(QuoteNode(accepts_output_grad)))))
    bindings = Dict{Symbol,Symbol}() # Map from variable names to node names
    # Generate statements that add nodes to the IR for each function argument
    for arg in args
        if arg.default != nothing
            error("Default argument values not supported in the static DSL.")
        end
        node = gensym(arg.name)
        push!(stmts, :($(esc(node)) = add_argument_node!(
            builder, name=$(QuoteNode(arg.name)), typ=$(QuoteNode(arg.typ)),
            compute_grad=$(QuoteNode(DSL_ARG_GRAD_ANNOTATION in arg.annotations)))))
        bindings[arg.name] = node
    end
    # Parse function body and add corresponding nodes to the IR
    parse_static_dsl_function_body!(stmts, bindings, body)
    push!(stmts, :(ir = build_ir(builder)))
    expr = gensym("gen_fn_defn")
    # Handle function annotations (caching Julia nodes by default)
    track_diffs = DSL_TRACK_DIFFS_ANNOTATION in annotations
    cache_julia_nodes = !(DSL_NO_JULIA_CACHE_ANNOTATION in annotations)
    options = StaticIRGenerativeFunctionOptions(track_diffs, cache_julia_nodes)
    # Generate statements that define the GFI, specialized to this function
    # NOTE: use the eval() for the user's module, not Gen
    push!(stmts, :(Core.@__doc__ $(esc(name)) = $(esc(:eval))(
        generate_generative_function(ir, $(QuoteNode(name)), $(QuoteNode(options))))))
    # Return the block of statements, which will be evaluated at compile time
    Expr(:block, stmts...)
end
