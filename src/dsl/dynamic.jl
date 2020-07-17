const DYNAMIC_DSL_TRACE = Symbol("@trace")

"Convert Argument structs to ASTs."
function arg_to_ast(arg::Argument)
    ast = esc(arg.name)
    if (arg.default != nothing)
        default = something(arg.default)
        ast = Expr(:kw, ast, esc(default))
    end
    ast
end

"Escape argument defaults (if present)."
function escape_default(arg)
    (arg.default == nothing ? nothing :
        Expr(:call, :Some, esc(something(arg.default))))
end

"Rewrites :gentrace and :genparam with their dynamic implementations."
function rewrite_dynamic_gen_exprs(expr)
    MacroTools.postwalk(expr) do e
        if MacroTools.@capture(e, ex_gentrace)
            return dynamic_trace_impl(ex)
        elseif MacroTools.@capture(e, ex_genparam)
            return dynamic_param_impl(ex)
        else
            return e
        end
    end
end

"Construct a dynamic Gen function."
function make_dynamic_gen_function(name, args, body, return_type, annotations)
    escaped_args = map(arg_to_ast, args)
    gf_args = [esc(state), escaped_args...]
    body = rewrite_dynamic_gen_exprs(body)

    julia_fn_name = gensym(name)
    julia_fn_defn = Expr(:function,
        Expr(:call, esc(julia_fn_name), gf_args...),
        esc(body))
    arg_types = map((arg) -> esc(arg.typ), args)
    arg_defaults = map(escape_default, args)
    has_argument_grads = map(
        (arg) -> (DSL_ARG_GRAD_ANNOTATION in arg.annotations), args)
    accepts_output_grad = DSL_RET_GRAD_ANNOTATION in annotations

    quote
        # first define the underlying Julia function
        $julia_fn_defn

        # now wrap it in a DynamicDSLFunction value
        Core.@__doc__ $(esc(name)) = DynamicDSLFunction(
            Type[$(arg_types...)],
            Union{Some{Any},Nothing}[$(arg_defaults...)],
            $(esc(julia_fn_name)),
            $has_argument_grads,
            $(esc(return_type)),
            $accepts_output_grad)
    end
end
