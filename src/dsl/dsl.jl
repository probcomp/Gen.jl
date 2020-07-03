import MacroTools

const DSL_STATIC_ANNOTATION = :static
const DSL_ARG_GRAD_ANNOTATION = :grad
const DSL_RET_GRAD_ANNOTATION = :grad
const DSL_TRACK_DIFFS_ANNOTATION = :diffs
const DSL_NO_JULIA_CACHE_ANNOTATION = :nojuliacache

# the set of macros we should not macroexpand before parsing
const DSL_MACROS = Set([Symbol("@trace"), Symbol("@param"), Symbol("@gen")])

struct Argument
    name::Symbol
    typ::Union{Symbol,Expr}
    annotations::Set{Symbol}
    default::Union{Some{Any}, Nothing}
end

Argument(name, typ) = Argument(name, typ, Set{Symbol}(), nothing)
Argument(name, typ, annotations) = Argument(name, typ, annotations, nothing)

function parse_annotations(annotations_expr)
    annotations = Set{Symbol}()
    if isa(annotations_expr, Symbol)
        push!(annotations, annotations_expr)
    elseif isa(annotations_expr, Expr) && annotations_expr.head == :tuple
        for annotation in annotations_expr.args
            if !isa(annotation, Symbol)
                error("syntax error in annotations_expr at $annotation")
            else
                push!(annotations, annotation)
            end
        end
    else
        error("syntax error in annotations at $annotations")
    end
    annotations
end

function parse_arg(expr)
    if isa(expr, Symbol)
        # x
        arg = Argument(expr, :Any)
    elseif isa(expr, Expr) && expr.head == :(::)
        # x::Int
        arg = Argument(expr.args[1], expr.args[2])
    elseif isa(expr, Expr) && expr.head == :kw
        # x::Int=1
        sub_arg = parse_arg(expr.args[1])
        default = Some(expr.args[2])
        arg = Argument(sub_arg.name, sub_arg.typ, Set{Symbol}(), default)
    elseif isa(expr, Expr) && expr.head == :call
        # (grad,foo)(x::Int)
        annotations_expr = expr.args[1]
        sub_arg = parse_arg(expr.args[2])
        annotations = parse_annotations(annotations_expr)
        arg = Argument(sub_arg.name, sub_arg.typ, annotations, sub_arg.default)
    else
        dump(expr)
        error("syntax error in gen function argument at $expr")
    end
    arg
end

include("dynamic.jl")
include("static.jl")

function address_from_expression(lhs)
    if lhs isa Symbol
        QuoteNode(lhs)
    else
        error("Syntax error: Only a variable or an address expression can appear on the lefthand side of a ~. Invalid left-hand side: $(lhs).")
    end
end

function desugar_tildes(expr)
    trace_ref = Symbol("@trace")
    line_num = LineNumberNode(1, :none)
    MacroTools.postwalk(expr) do e
        # Replace with globally referenced macrocalls
        if MacroTools.@capture(e, {*} ~ rhs_)
            Expr(:macrocall, trace_ref, line_num, rhs)
        elseif MacroTools.@capture(e, {addr_} ~ rhs_)
            Expr(:macrocall, trace_ref, line_num, rhs, addr)
        elseif MacroTools.@capture(e, lhs_ ~ rhs_)
            addr = address_from_expression(lhs)
            Expr(:(=), lhs, Expr(:macrocall, trace_ref, line_num, rhs, addr))
        else
            e
        end
    end
end

# Resolves Gen macros to globally referenced macro calls;
# fully macroexpands all macros Gen does not recognize
function resolve_or_expand_macros(expr, __module__)
    MacroTools.postwalk(expr) do e
        if MacroTools.@capture(e, @namespace_.m_(args__) | @m_(args__)) # for any macro expression...
            # if this is already a GlobalRef, the macro's name is stored in `m.name`
            macroname = m isa GlobalRef ? m.name : m

            if macroname in DSL_MACROS # if this is a Gen macro name (ie. @trace or @param)
                # The macro is either from the local module `__module__` or explicitly specifies a `namespace`
                if namespace === nothing
                    mod = __module__
                else
                    mod =__module__.eval(namespace)
                end

                # if the macro is defined in the given module, and refers to the Gen macro
                if isdefined(mod, macroname) && getfield(mod, macroname) == getfield(@__MODULE__, macroname)
                    macro_ref = GlobalRef(@__MODULE__, macroname)
                    line_num = e.args[2]
                    return Expr(:macrocall, macro_ref, line_num, args...)
                end
            end
            # if we get here, then this is not a Gen macro, so macroexpand it
            return macroexpand(__module__, e)
        else # not a macro
            return e
        end
    end
end

function extract_quoted_exprs(expr)
    quoted_exprs = []
    expr = MacroTools.prewalk(expr) do e
        if MacroTools.@capture(e, :(quoted_)) && !isa(quoted, Symbol)
            push!(quoted_exprs, e)
            Expr(:placeholder, length(quoted_exprs))
        else
            e
        end
    end
    return expr, quoted_exprs
end

function insert_quoted_exprs(expr, quoted_exprs)
    expr = MacroTools.prewalk(expr) do e
        if MacroTools.@capture(e, p_placeholder)
            idx = p.args[1]
            quoted_exprs[idx]
        else
            e
        end
    end
    return expr
end

function preprocess_body(expr, __module__)
    # Protect quoted expressions from pre-processing by extracting them
    expr, quoted_exprs = extract_quoted_exprs(expr)
    # Desugar tilde calls to globally referenced @trace calls
    expr = desugar_tildes(expr)
    # Resolve Gen macros to GlobalRefs for consistent downstream parsing,
    # and expand macros which are not from Gen
    expr = resolve_or_expand_macros(expr, __module__)
    # Reinsert quoted expressions after pre-processing
    expr = insert_quoted_exprs(expr, quoted_exprs)
    return expr
end

function parse_gen_function(ast, annotations, __module__)
    ast = MacroTools.longdef(ast)
    if ast.head != :function
        error("syntax error at $ast in $(ast.head)")
    end
    if length(ast.args) != 2
        error("syntax error at $ast in $(ast.args)")
    end
    signature = ast.args[1]
    if signature.head == :(::)
        (call_signature, return_type) = signature.args
    elseif signature.head == :call
        (call_signature, return_type) = (signature, :Any)
    else
        error("syntax error at $(signature)")
    end
    body = preprocess_body(ast.args[2], __module__)
    name = call_signature.args[1]
    args = map(parse_arg, call_signature.args[2:end])
    static = DSL_STATIC_ANNOTATION in annotations
    if static
        make_static_gen_function(name, args, body, return_type, annotations)
    else
        make_dynamic_gen_function(name, args, body, return_type, annotations)
    end
end

macro gen(annotations_expr, ast::Expr)
    # parse the annotations
    annotations = parse_annotations(annotations_expr)
    # parse the function definition
    parse_gen_function(ast, annotations, __module__)
end

macro gen(ast::Expr)
    parse_gen_function(ast, Set{Symbol}(), __module__)
end
