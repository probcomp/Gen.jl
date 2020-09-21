function choice_gradients(trace::VectorTrace{UnfoldType,T,U}, selection::Selection,
                        retval_grad) where {T,U}
    error("Not implemented")
end

function accumulate_param_gradients!(trace::VectorTrace{UnfoldType,T,U}, retval_grad) where {T,U}
    args = get_args(trace)
    (len, init_state, params) = unpack_args(args)

    kernel_has_grads = has_argument_grads(trace.gen_fn.kernel)
    if kernel_has_grads[1]
        error("Cannot differentiate with respect to index in unfold")
    end
    if kernel_has_grads[2]
        # backpropagation through the state not yet implemented
        error("Not implemented")
    end
    params_has_grad = kernel_has_grads[3:end]

    params_grad = Vector(undef, length(params_has_grad))
    for (i, has_grad) in enumerate(params_has_grad)
        if has_grad
            params_grad[i] = Vector(undef, len)
        else
            params_grad[i] = nothing
        end
    end

    for key=1:len
        subtrace = trace.subtraces[key]
        kernel_retval_grad = (retval_grad == nothing) ? nothing : retval_grad[key]
        kernel_arg_grad::Tuple = accumulate_param_gradients!(subtrace, kernel_retval_grad)
        @assert kernel_arg_grad[1] == nothing
        @assert kernel_arg_grad[2] == nothing
        for (i, (grad, has_grad)) in enumerate(zip(kernel_arg_grad[3:end], params_has_grad))
            if has_grad
                params_grad[i][key] = grad
            end
        end
    end
    (nothing, nothing, params_grad...)
end
