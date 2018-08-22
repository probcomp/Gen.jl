###################
# plate generator # 
###################

"""
Generator that makes many independent application of a kernel generator,
similar to 'map'.  The arguments are a tuple of vectors, each of of length N,
where N is the nubmer of applications of the kernel.
"""
struct PlateOfDist{T} <: Generator{PersistentVector{T},VectorDistTrace{T}}
    kernel::Distribution{T}
end

# TODO
#accepts_output_grad(plate::Plate) = false
#has_argument_grads(plate::Plate) = has_argument_grads(plate.kernel)

function plate(kernel::Distribution{T}) where {T}
    PlateOfDist{T}(kernel)
end

function get_static_argument_types(plate::PlateOfDist)
    [Expr(:curly, :Vector, typ) for typ in get_static_argument_types(plate.kernel)]
end

function simulate(gen::PlateOfDist{T}, args, read_trace=nothing) where {T}
    len = length(args[1])
    values = Vector{T}(undef, len)
    score = 0.
    for key=1:len
        kernel_args = get_args_for_key(args, key)
        values[key] = random(gen.kernel, kernel_args...)
        score += logpdf(gen.kernel, values[key], kernel_args...)
    end
    persist_vec = Gen.PersistentVector{T}(values)
    call = CallRecord(score, persist_vec, args)
    Gen.VectorDistTrace{T}(persist_vec, call)
end

function generate(gen::PlateOfDist{T}, args, constraints, read_trace=nothing) where {T}
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
    persist_vec = Gen.PersistentVector{T}(values)
    call = CallRecord(score, persist_vec, args)
    trace = Gen.VectorDistTrace{T}(persist_vec, call)
    (trace, weight)
end

#function update(gen::PlateOfDist{T}, new_args, args_change::NoChange, trace::VectorDistTrace{T}, constraints, read_trace=nothing) where {T}
    #to_visit = Set{Int}()
    #_update(gen, new_args, trace, constraints, read_trace, to_visit)
#end

function update(gen::PlateOfDist{T}, new_args, args_change::Nothing, trace::VectorDistTrace{T}, constraints, read_trace=nothing) where {T}
    (new_length, prev_length) = get_prev_and_new_lengths(new_args, trace)
    @assert new_length == prev_length
    # TODO handle increases or decreaeses int e length 

    prev_args = get_call_record(trace).args
    score = get_call_record(trace).score
    weight = 0.
    values = trace.values
    for key=1:new_length
        prev_value = values[key]
        prev_kernel_args = get_args_for_key(prev_args, key)
        kernel_args = get_args_for_key(new_args, key)
        is_constrained = has_leaf_node(constraints, key)
        if is_constrained
            @assert false
            value = get_leaf_node(constraints, key)
            values = assoc(values, key, value)
        else
            value = prev_value
        end
        lpdf = logpdf(gen.kernel, value, kernel_args...)
        prev_lpdf = logpdf(gen.kernel, prev_value, prev_kernel_args...)
        weight += lpdf - prev_lpdf
        score += lpdf - prev_lpdf
    end
    # TODO also check that there are no extra constraints
    # TODO handle retchange
    call = CallRecord(score, values, new_args)
    new_trace = Gen.VectorDistTrace{T}(values, call)
    (new_trace, weight, EmptyChoiceTrie(), nothing)
end

#function _update(gen::PlateOfDist{T}, args, prev_trace::VectorDistTrace{T}, constraints, read_trace, to_visit) where {T}
#
#end

export plate
