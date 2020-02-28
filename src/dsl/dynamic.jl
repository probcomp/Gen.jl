const DYNAMIC_DSL_TRACE = Symbol("@trace")

function arg_to_ast(arg::Argument)
    ast = esc(arg.name)
    if (arg.default != nothing)
        default = something(arg.default)
        ast = Expr(:kw, ast, esc(default))
    end
    ast
end

function make_dynamic_gen_function(name, args, body, return_type, annotations)
    escaped_args = map(arg_to_ast, args)
    gf_args = [esc(state), escaped_args...]

    julia_fn_name = gensym(name)
    julia_fn_defn = Expr(:function,
        Expr(:call, esc(julia_fn_name), gf_args...),
        esc(body))
    arg_types = map((arg) -> esc(arg.typ), args)
    arg_defaults = map((arg) -> esc(arg.default), args)
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
