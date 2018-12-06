#################################
# map of distribution generator # 
#################################

struct MapDist{T} <: GenerativeFunction{PersistentVector{T},VectorDistTrace{T}}
    kernel::Distribution{T}
end

export MapDist

accepts_output_grad(map_gf::MapDist) = false # TODO
has_argument_grads(map_gf::MapDist) = has_argument_grads(map_gf.kernel)

function simulate(gen::MapDist{T}, args) where {T}
    len = length(args[1])
    values = Vector{T}(undef, len)
    score = 0.
    for key=1:len
        kernel_args = get_args_for_key(args, key)
        values[key] = random(gen.kernel, kernel_args...)
        score += logpdf(gen.kernel, values[key], kernel_args...)
    end
    persist_values = PersistentVector{T}(values)
    VectorDistTrace(persist_values, args, score, len)
end

function generate(gen::MapDist{T}, args, constraints) where {T}
    len = length(args[1])
    values = Vector{T}(undef, len)
    score = 0.
    weight = 0.
    for key=1:len
        kernel_args = get_args_for_key(args, key)
        is_constrained = has_leaf_node(constraints, key)
        if is_constrained
            value = get_leaf_node(constraints, key)
        else
            value = random(gen.kernel, kernel_args...)
        end
        values[key] = value
        lpdf = logpdf(gen.kernel, value, kernel_args...)
        if is_constrained
            weight += lpdf
        end
        score += lpdf
    end
    # TODO also check that there are no extra constraints
    persist_values = Gen.PersistentVector{T}(values)
    trace = VectorDistTrace(persist_values, args, score, len)
    (trace, weight)
end

function assess(gen::MapDist{T}, args, constraints) where {T}
    len = length(args[1])
    values = Vector{T}(undef, len)
    score = 0.
    for key=1:len
        kernel_args = get_args_for_key(args, key)
        value = get_leaf_node(constraints, key)
        values[key] = value
        lpdf = logpdf(gen.kernel, value, kernel_args...)
        score += lpdf
    end
    # TODO also check that there are no extra constraints
    persist_values = Gen.PersistentVector{T}(values)
    trace = VectorDistTrace(persist_values, args, score, len)
    trace
end

function update(gen::MapDist{T}, new_args, argdiff::Union{NoArgDiff,UnknownArgDiff},
                trace::VectorDistTrace{T}, constraints) where {T}
    (new_length, prev_length) = get_prev_and_new_lengths(new_args, trace)
    @assert new_length == prev_length
    # TODO handle increases or decreaeses int e length 

    prev_args = get_call_record(trace).args
    score = get_call_record(trace).score
    weight = 0.
    values = trace.values
    discard = DynamicAssignment()
    for key=1:new_length
        prev_value = values[key]
        prev_kernel_args = get_args_for_key(prev_args, key)
        kernel_args = get_args_for_key(new_args, key)
        is_constrained = has_leaf_node(constraints, key)
        if is_constrained
            value = get_leaf_node(constraints, key)
            values = assoc(values, key, value)
            set_leaf_node!(discard, key, prev_value)
        else
            value = prev_value
        end
        lpdf = logpdf(gen.kernel, value, kernel_args...)
        prev_lpdf = logpdf(gen.kernel, prev_value, prev_kernel_args...)
        weight += lpdf - prev_lpdf
        score += lpdf - prev_lpdf
    end
    # TODO also check that there are no extra constraints
    new_trace = VectorDistTrace(values, new_args, score, new_length)
    (new_trace, weight, discard, DefaultRetDiff())
end

function backprop_trace(gen::MapDist{T}, trace::VectorDistTrace{T},
                        selection::EmptyAddressSet, retval_grad) where {T}
    call = get_call_record(trace)
    args = call.args
    n_args = length(args)
    len = length(args[1])
    
    has_grads = has_argument_grads(gen.kernel)
    arg_grad = Vector(undef, n_args)
    for (i, has_grad) in enumerate(has_grads)
        if has_grad
            arg_grad[i] = Vector(undef, len)
        else
            arg_grad[i] = nothing
        end
    end
    
    for key=1:len
        value = trace.values[key]
        kernel_args = get_args_for_key(args, key)
        kernel_grads = logpdf_grad(gen.kernel, value, kernel_args...)
        kernel_arg_grad = kernel_grads[2:end]
        for (i, grad, has_grad) in zip(1:n_args, kernel_arg_grad, has_grads)
            if has_grad
                arg_grad[i][key] = grad
            end
        end
    end
    value_trie = EmptyAssignment()
    gradient_trie = EmptyAssignment()
    ((arg_grad...,), value_trie, gradient_trie)
end

function backprop_trace(gen::MapDist{T}, trace::VectorDistTrace{T},
                        selection, retval_grad) where {T}
    error("Not yet implemented!")
end
