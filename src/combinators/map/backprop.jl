function choice_gradients(trace::VectorTrace{MapType,T,U}, selection::AddressSet,
                        retval_grad) where {T,U}

    args = get_args(trace)
    n_args = length(args)
    len = length(args[1])
    
    has_grads = has_argument_grads(trace.gen_fn.kernel)
    arg_grad = Vector(undef, n_args)
    for (i, has_grad) in enumerate(has_grads)
        if has_grad
            arg_grad[i] = Vector(undef, len)
        else
            arg_grad[i] = nothing
        end
    end
    
    value_assmt = DynamicAssignment()
    gradient_assmt = DynamicAssignment()
    
    for key=1:len
        subtrace = trace.subtraces[key]
        if has_internal_node(selection, key)
            sub_selection = get_internal_node(selection, key)
        else
            sub_selection = EmptyAddressSet()
        end
        kernel_retval_grad = (retval_grad == nothing) ? nothing : retval_grad[key]
        (kernel_arg_grad::Tuple, kernel_value_assmt, kernel_gradient_assmt) = choice_gradients(
            subtrace, sub_selection, kernel_retval_grad)
        set_subassmt!(value_assmt, key, kernel_value_assmt)
        set_subassmt!(gradient_assmt, key, kernel_gradient_assmt)
        for (i, grad, has_grad) in zip(1:n_args, kernel_arg_grad, has_grads)
            if has_grad
                arg_grad[i][key] = grad
            end
        end
    end
    ((arg_grad...,), value_assmt, gradient_assmt)
end

function accumulate_param_gradients!(trace::VectorTrace{MapType,T,U}, retval_grad) where {T,U}

    args = get_args(trace)
    n_args = length(args)
    len = length(args[1])
    
    has_grads = has_argument_grads(trace.gen_fn.kernel)
    arg_grad = Vector(undef, n_args)
    for (i, has_grad) in enumerate(has_grads)
        if has_grad
            arg_grad[i] = Vector(undef, len)
        else
            arg_grad[i] = nothing
        end
    end
    
    for key=1:len
        subtrace = trace.subtraces[key]
        kernel_retval_grad = (retval_grad == nothing) ? nothing : retval_grad[key]
        kernel_arg_grad::Tuple = accumulate_param_gradients!(subtrace, kernel_retval_grad)
        for (i, grad, has_grad) in zip(1:n_args, kernel_arg_grad, has_grads)
            if has_grad
                arg_grad[i][key] = grad
            end
        end
    end
    (arg_grad...,)
end
