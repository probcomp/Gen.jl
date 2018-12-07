function backprop_params(gen::HigherOrderMap, trace::VectorTrace{T,U}, retval_grad) where {T,U}
    (kernel, first_order_trace) = extract_plate_first_order(trace)
    backprop_params(Map(kernel), first_order_trace, retval_grad)
end

function backprop_params(gen::Map{T,U}, trace::VectorTrace{T,U}, retval_grad) where {T,U}

    call = get_call_record(trace)
    args = call.args
    n_args = length(args)
    len = length(args[1])
    
    has_grads = has_argument_grads(gen.kernel)
    arg_grad = Vector(n_args)
    for (i, has_grad) in enumerate(has_grads)
        if has_grad
            arg_grad[i] = Vector(len)
        else
            arg_grad[i] = nothing
        end
    end
    for key=1:len
        subtrace = trace.subtraces[key]
        kernel_retval_grad = (retval_grad == nothing) ? nothing : retval_grad[key]
        kernel_arg_grad::Tuple = backprop_params(gen.kernel, subtrace, kernel_retval_grad)
        for (i, grad, has_grad) in zip(1:n_args, kernel_arg_grad, has_grads)
            if has_grad
                arg_grad[i][key] = grad
            end
        end
    end
    (arg_grad...)
end
