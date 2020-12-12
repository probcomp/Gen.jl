function choice_gradients(trace::VectorTrace{UnfoldType,T,U}, selection::Selection,
                        retval_grad) where {T,U}
    error("Not implemented")
end

# Utility method for state gradient accumulation.
@inline fold_sum(::Tuple{Nothing, Nothing}) = nothing
@inline fold_sum(x::Tuple) = sum(x)

function accumulate_param_gradients!(trace::VectorTrace{UnfoldType,T,U}, retval_grad) where {T,U}
    args = get_args(trace)
    (len, init_state, params) = unpack_args(args)

    kernel_has_grads = has_argument_grads(trace.gen_fn.kernel)
    if kernel_has_grads[1]
        error("Cannot differentiate with respect to index in unfold")
    end
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
            state_retval_grad = (key == len) ? nothing : kernel_arg_grads[2]
            kernel_arg_grads = map((kernel_retval_grad, state_retval_grad)) do a
                accumulate_param_gradients!(subtrace, a)
            end # run two passes - first for retval, second for state.
            kernel_arg_grads = Tuple(map(x -> fold_sum(x), zip(kernel_arg_grads...))) # collapse grads
            @assert kernel_arg_grads[1] == nothing
            for (i, (grad, has_grad)) in enumerate(zip(kernel_arg_grads[3:end], params_has_grad))
                if has_grad
                    params_grad[i][key] = grad
                end
            end
        
        else # ignore state grad
            kernel_arg_grads = accumulate_param_gradients!(subtrace, kernel_retval_grad)
            @assert kernel_arg_grads[1] == nothing
            @assert kernel_arg_grads[2] == nothing
            for (i, (grad, has_grad)) in enumerate(zip(kernel_arg_grads[3:end], params_has_grad))
                if has_grad
                    params_grad[i][key] = grad
                end
            end
        end
    end
    (nothing, kernel_arg_grads[2], params_grad...)
end
