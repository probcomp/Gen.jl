function choice_gradients(trace::VectorTrace{MapType,T,U}, selection::Selection,
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

    value_choices = choicemap()
    gradient_choices = choicemap()

    for key=1:len
        subtrace = trace.subtraces[key]
        sub_selection = selection[key]
        kernel_retval_grad = (retval_grad == nothing) ? nothing : retval_grad[key]
        (kernel_arg_grad::Tuple, kernel_value_choices, kernel_gradient_choices) = choice_gradients(
            subtrace, sub_selection, kernel_retval_grad)
        set_submap!(value_choices, key, kernel_value_choices)
        set_submap!(gradient_choices, key, kernel_gradient_choices)
        for (i, grad, has_grad) in zip(1:n_args, kernel_arg_grad, has_grads)
            if has_grad
                arg_grad[i][key] = grad
            end
        end
    end
    ((arg_grad...,), value_choices, gradient_choices)
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
