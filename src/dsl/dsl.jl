import MacroTools

const DSL_STATIC_ANNOTATION = :static
const DSL_ARG_GRAD_ANNOTATION = :grad
const DSL_RET_GRAD_ANNOTATION = :grad
const DSL_TRACK_DIFFS_ANNOTATION = :diffs
const DSL_NO_JULIA_CACHE_ANNOTATION = :nojuliacache

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
    MacroTools.postwalk(expr) do e
        # Expand the `@trace` macro as defined in this module (even if the caller
        # doesn't have an analogous macro in their module), and leave the
        # remaining macros to be expanded in the caller's scope.
        if MacroTools.@capture(e, {*} ~ rhs_)
            macroexpand(@__MODULE__, :(@trace($rhs)), recursive=false)
        elseif MacroTools.@capture(e, {addr_} ~ rhs_)
            macroexpand(@__MODULE__, :(@trace($rhs, $(addr))), recursive=false)
        elseif MacroTools.@capture(e, lhs_ ~ rhs_)
            addr_expr = address_from_expression(lhs)
            macroexpand(@__MODULE__, :($lhs = @trace($rhs, $(addr_expr))),
                        recursive=false)
        else
            e
        end
    end
end

function parse_gen_function(ast, annotations)
    ast = MacroTools.longdef(ast)
    if ast.head != :function
        error("syntax error at $ast in $(ast.head)")
    end
    if length(ast.args) != 2
        error("syntax error at $ast in $(ast.args)")
    end
    signature = ast.args[1]
    body = desugar_tildes(ast.args[2])
    if signature.head == :(::)
        (call_signature, return_type) = signature.args
    elseif signature.head == :call
        (call_signature, return_type) = (signature, :Any)
    else
        error("syntax error at $(signature)")
    end
    name = call_signature.args[1]
    args = map(parse_arg, call_signature.args[2:end])
    static = DSL_STATIC_ANNOTATION in annotations
    if static
        make_static_gen_function(name, args, body, return_type, annotations)
    else
        make_dynamic_gen_function(name, args, body, return_type, annotations)
    end
end

macro gen(annotations_expr, ast)

    # parse the annotations
    annotations = parse_annotations(annotations_expr)

    # parse the function definition
    parse_gen_function(ast, annotations)
end

macro gen(ast)
    parse_gen_function(ast, Set{Symbol}())
end
