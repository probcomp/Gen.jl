const DYNAMIC_DSL_ADDR = Symbol("@addr")
const DYNAMIC_DSL_DIFF = Symbol("@diff")

transform_body_for_non_update(ast) = ast

function transform_body_for_non_update(ast::Expr)
    @assert !(ast.head == :macrocall && ast.args[1] == DYNAMIC_DSL_DIFF)

    # remove the argdiff argument from @addr expressions
    if ast.head == :macrocall && ast.args[1] == DYNAMIC_DSL_ADDR
        if length(ast.args) == 5
            @assert isa(ast.args[2], LineNumberNode)
            # remove the last argument
            ast = Expr(ast.head, ast.args[1:4]...)
        end
    end

    # remove any @diff sub-expressions, and recurse
    new_args = Vector()
    for arg in ast.args
        if !(isa(arg, Expr) && arg.head == :macrocall && arg.args[1] == DYNAMIC_DSL_DIFF)
            push!(new_args, transform_body_for_non_update(arg))
        end
    end
    Expr(ast.head, new_args...)
end

transform_body_for_update(ast, in_diff) = ast

function transform_body_for_update(ast::Expr, in_diff::Bool)
    @assert !(ast.head == :macrocall && ast.args[1] == DYNAMIC_DSL_DIFF)
    new_args = Vector()
    for arg in ast.args
        if isa(arg, Expr) && arg.head == :macrocall && arg.args[1] == DYNAMIC_DSL_DIFF
            if in_diff
                error("Got nested @diff at: $arg")
            end
            if length(arg.args) == 2
                push!(new_args, transform_body_for_update(arg.args[2], true))
            elseif length(arg.args) == 3 && isa(arg.args[2], LineNumberNode)
                push!(new_args, transform_body_for_update(arg.args[3], true))
            else
                error("@diff must have just one sub-expression, got $arg")
            end
        else
            push!(new_args, arg)
        end
    end
    Expr(ast.head, new_args...)
end

function make_dynamic_gen_function(name, args, body, return_type, annotations)
    # julia function definition (for a non-update behavior)
    escaped_args = map((arg) -> esc(arg.name), args)
    gf_args = [esc(state), escaped_args...]
    julia_fn_defn = Expr(:function,
        Expr(:tuple, gf_args...),
        esc(transform_body_for_non_update(body)))

    # julia function definition for update function
    update_julia_fn_defn = Expr(:function,
        Expr(:tuple, gf_args...),
        esc(transform_body_for_update(body, false)))

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
                julia_fn_defn, update_julia_fn_defn,
                has_argument_grads, return_type,
                accepts_output_grad)))
end
