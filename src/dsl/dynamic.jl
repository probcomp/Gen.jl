const DYNAMIC_DSL_TRACE = Symbol("@trace")

function make_dynamic_gen_function(name, args, body, return_type, annotations)
    escaped_args = map((arg) -> esc(arg.name), args)
    gf_args = [esc(state), escaped_args...]
    julia_fn_defn = Expr(:function,
        Expr(:tuple, gf_args...),
        esc(body))

    # create generator and assign it a name
    arg_types = map((arg) -> esc(arg.typ), args)
    has_argument_grads = map(
        (arg) -> (DSL_ARG_GRAD_ANNOTATION in arg.annotations), args)
    accepts_output_grad = DSL_RET_GRAD_ANNOTATION in annotations
    Expr(:block,
        Expr(:(=), 
            esc(name),
            Expr(:call, :DynamicDSLFunction,
                quote Type[$(arg_types...)] end,
                julia_fn_defn,
                has_argument_grads, return_type,
                accepts_output_grad)))
end
