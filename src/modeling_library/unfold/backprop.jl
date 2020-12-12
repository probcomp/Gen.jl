# Utility method for state gradient accumulation.
@inline fold_sum(::Nothing, ::Nothing) = nothing
@inline fold_sum(a::A, ::Nothing) where A = a
@inline fold_sum(::Nothing, a::A) where A = a

function choice_gradients(trace::VectorTrace{UnfoldType,T,U}, selection::Selection, retval_grad) where {T,U}
    kernel_has_grads = has_argument_grads(trace.gen_fn.kernel)
    if kernel_has_grads[1]
        error("Cannot differentiate with respect to index in unfold")
    end

    args = get_args(trace)
    (len, init_state, params) = unpack_args(args)
    state_has_grad = kernel_has_grads[2]
    params_has_grad = kernel_has_grads[3 : end]
    value_choices = choicemap()
    gradient_choices = choicemap()
    params_grad = Vector(undef, length(params_has_grad))
    for (i, has_grad) in enumerate(params_has_grad)
        if has_grad
            params_grad[i] = Vector(undef, len)
        else
            params_grad[i] = nothing
        end
    end

    local kernel_arg_grads :: Tuple = (nothing, nothing)
    for key = len : -1 : 1 # walks backward over chain
        subtrace = trace.subtraces[key]
        subselection = get_subselection(selection, key)
        kernel_retval_grad = (retval_grad == nothing) ? nothing : retval_grad[key]
        if state_has_grad
            kernel_retval_grad = fold_sum(kernel_retval_grad, kernel_arg_grads[2])
        end
        kernel_arg_grads, kernel_value_choices, kernel_gradient_choices = choice_gradients(subtrace, subselection, kernel_retval_grad)
        set_submap!(value_choices, key, kernel_value_choices)
        set_submap!(gradient_choices, key, kernel_gradient_choices)
        @assert kernel_arg_grads[1] == nothing
        state_has_grad || @assert kernel_arg_grads[2] == nothing
        for (i, (grad, has_grad)) in enumerate(zip(kernel_arg_grads[3:end], params_has_grad))
            if has_grad
                params_grad[i][key] = grad
            end
        end
    end
    ((nothing, kernel_arg_grads[2], params_grad...), value_choices, gradient_choices)
end

function accumulate_param_gradients!(trace::VectorTrace{UnfoldType,T,U}, retval_grad) where {T,U}
    kernel_has_grads = has_argument_grads(trace.gen_fn.kernel)
    if kernel_has_grads[1]
        error("Cannot differentiate with respect to index in unfold")
    end

    args = get_args(trace)
    (len, init_state, params) = unpack_args(args)
    state_has_grad = kernel_has_grads[2]
    params_has_grad = kernel_has_grads[3:end]
    params_grad = Vector(undef, length(params_has_grad))
    for (i, has_grad) in enumerate(params_has_grad)
        if has_grad
            params_grad[i] = Vector(undef, len)
        else
            params_grad[i] = nothing
        end
    end

    local kernel_arg_grads::Tuple = (nothing, nothing) # initialize for (len index, state)
    for key = len : -1 : 1 # walks backward over chain
        subtrace = trace.subtraces[key]
        kernel_retval_grad = (retval_grad == nothing) ? nothing : retval_grad[key]
        if state_has_grad
            kernel_retval_grad = fold_sum(kernel_retval_grad, kernel_arg_grads[2])
        end
        kernel_arg_grads = accumulate_param_gradients!(subtrace, kernel_retval_grad)
        @assert kernel_arg_grads[1] == nothing
        state_has_grad || @assert kernel_arg_grads[2] == nothing
        for (i, (grad, has_grad)) in enumerate(zip(kernel_arg_grads[3:end], params_has_grad))
            if has_grad
                params_grad[i][key] = grad
            end
        end
    end
    (nothing, kernel_arg_grads[2], params_grad...)
end
