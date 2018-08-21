function backprop_trace(gen::Plate{T,U}, trace::VectorTrace{T,U}, selection::AddressSet, retval_grad, read_trace=nothing) where {T,U}

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
    
    value_trie = HomogenousTrie{Any,Float64}()
    gradient_trie = HomogenousTrie{Any,Float64}()
    
    for key=1:len
        subtrace = trace.subtraces[key]
        sub_selection = selection[key]
        kernel_retval_grad = (retval_grad == nothing) ? nothing : retval_grad[key]
        (kernel_arg_grad::Tuple, kernel_value_trie, kernel_gradient_trie) = backprop_trace(
            gen.kernel, subtrace, sub_selection, kernel_retval_grad, read_trace)
        set_internal_node!(value_trie, key, kernel_value_trie)
        set_internal_node!(gradient_trie, key, kernel_gradient_trie)
        for (i, grad, has_grad) in zip(1:n_args, kernel_arg_grad, has_grads)
            if has_grad
                arg_grad[i][key] = grad
            end
        end
    end
    ((arg_grad...), value_trie, gradient_trie)
end
