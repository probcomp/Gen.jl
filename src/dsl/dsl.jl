export @gen, @param, @trace

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

function desugar_tildes(expr)
    MacroTools.postwalk(expr) do e
        # Replace tilde statements with :gentrace expressions
        if MacroTools.@capture(e, {*} ~ rhs_call)
            Expr(:gentrace, rhs, nothing)
        elseif MacroTools.@capture(e, {addr_} ~ rhs_call)
            Expr(:gentrace, rhs, Some(addr))
        elseif MacroTools.@capture(e, lhs_Symbol ~ rhs_call)
            addr = QuoteNode(lhs)
            Expr(:(=), lhs, Expr(:gentrace, rhs, Some(addr)))
        elseif MacroTools.@capture(e, lhs_ ~ rhs_call)
            error("Syntax error: Invalid left-hand side: $(e)." *
                  "Only a variable or address can appear on the left of a `~`.")
        elseif MacroTools.@capture(e, lhs_ ~ rhs_)
            error("Syntax error: Invalid right-hand side in: $(e)")
        else
            e
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
    # Expand all macros relative to the calling module
    expr = macroexpand(__module__, expr)
    # Protect quoted expressions from pre-processing by extracting them
    expr, quoted_exprs = extract_quoted_exprs(expr)
    # Desugar tilde calls to :gentrace expressions
    expr = desugar_tildes(expr)
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

macro trace(expr::Expr)
    return Expr(:gentrace, esc(expr), nothing)
end

macro trace(expr::Expr, addr)
    return Expr(:gentrace, esc(expr), Some(addr))
end

macro param(expr::Expr)
    if (expr.head != :(::)) error("Syntax in error in @param at $(expr)") end
    name, type = expr.args
    return Expr(:genparam, esc(name), esc(type))
end

macro param(sym::Symbol)
    return Expr(:genparam, esc(sym), esc(:Any))
end
