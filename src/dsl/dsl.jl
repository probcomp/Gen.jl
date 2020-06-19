import MacroTools

const DSL_STATIC_ANNOTATION = :static
const DSL_ARG_GRAD_ANNOTATION = :grad
const DSL_RET_GRAD_ANNOTATION = :grad
const DSL_TRACK_DIFFS_ANNOTATION = :diffs
const DSL_NO_JULIA_CACHE_ANNOTATION = :nojuliacache
const DSL_MACROS = Set([Symbol("@trace"), Symbol("@param")])

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
    trace_ref = GlobalRef(@__MODULE__, Symbol("@trace"))
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

function resolve_gen_macros(expr, __module__)
    MacroTools.postwalk(expr) do e
        # Resolve Gen macros to globally referenced macrocalls
        if (MacroTools.@capture(e, @namespace_.m_(args__)) &&
            m in DSL_MACROS && __module__.eval(namespace) == @__MODULE__)
            macro_ref = GlobalRef(@__MODULE__, m)
            line_num = e.args[2]
            Expr(:macrocall, macro_ref, line_num, args...)
        elseif (MacroTools.@capture(e, @m_(args__)) &&
                m in DSL_MACROS && __module__ == @__MODULE__)
            macro_ref = GlobalRef(@__MODULE__, m)
            line_num = e.args[2]
            Expr(:macrocall, macro_ref, line_num, args...)
        else
            e
        end
    end
end

function preprocess_body(expr, __module__)
    expr = desugar_tildes(expr)
    expr = resolve_gen_macros(expr, __module__)
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
