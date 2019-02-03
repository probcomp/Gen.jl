mutable struct MapForceUpdateState{T,U}
    weight::Float64
    score::Float64
    noise::Float64
    subtraces::PersistentVector{U}
    retval::PersistentVector{T}
    discard::DynamicAssignment
    num_nonempty::Int
    isdiff_retdiffs::Dict{Int,Any}
end

function process_retained!(gen_fn::Map{T,U}, args::Tuple,
                           assmt::Assignment, key::Int, kernel_argdiff,
                           state::MapForceUpdateState{T,U}) where {T,U}
    local subtrace::U
    local prev_subtrace::U
    local retval::T

    subassmt = get_subassmt(assmt, key)
    kernel_args = get_args_for_key(args, key)

    # get new subtrace with recursive call to update()
    prev_subtrace = state.subtraces[key]
    (subtrace, weight, subretdiff, discard) = update(
        prev_subtrace, kernel_args, kernel_argdiff, subassmt)

    # retrieve retdiff
    if !isnodiff(subretdiff)
        state.isdiff_retdiffs[key] = subretdiff
    end

    # update state
    state.weight += weight
    set_subassmt!(state.discard, key, discard)
    state.score += (get_score(subtrace) - get_score(prev_subtrace))
    state.noise += (project(subtrace, EmptyAddressSet()) - project(subtrace, EmptyAddressSet()))
    state.subtraces = assoc(state.subtraces, key, subtrace)
    retval = get_retval(subtrace)
    state.retval = assoc(state.retval, key, retval)
    subtrace_empty = isempty(get_assmt(subtrace))
    prev_subtrace_empty = isempty(get_assmt(prev_subtrace))
    if !subtrace_empty && prev_subtrace_empty
        state.num_nonempty += 1
    elseif subtrace_empty && !prev_subtrace_empty
        state.num_nonempty -= 1
    end
end

function process_new!(gen_fn::Map{T,U}, args::Tuple, assmt, key::Int,
                      state::MapForceUpdateState{T,U}) where {T,U}
    local subtrace::U
    local retval::T

    subassmt = get_subassmt(assmt, key)
    kernel_args = get_args_for_key(args, key)

    # get subtrace and weight
    (subtrace, weight) = generate(gen_fn.kernel, kernel_args, subassmt)

    # update state
    state.weight += weight
    state.score += get_score(subtrace)
    retval = get_retval(subtrace)
    @assert key > length(state.subtraces)
    state.subtraces = push(state.subtraces, subtrace)
    state.retval = push(state.retval, retval)
    @assert length(state.subtraces) == key
    if !isempty(get_assmt(subtrace))
        state.num_nonempty += 1
    end
end


function update(trace::VectorTrace{MapType,T,U}, args::Tuple, argdiff,
                assmt::Assignment) where {T,U}
    gen_fn = trace.gen_fn
    (new_length, prev_length) = get_prev_and_new_lengths(args, trace)
    retained_and_constrained = get_retained_and_constrained(assmt, prev_length, new_length)

    # handle removed applications
    (discard, num_nonempty, score_decrement, noise_decrement) = vector_update_delete(
        new_length, prev_length, trace)
    (subtraces, retval) = vector_remove_deleted_applications(
        trace.subtraces, trace.retval, prev_length, new_length)
    score = trace.score - score_decrement
    noise = trace.noise - noise_decrement
    
    # handle retained and new applications
    state = MapForceUpdateState{T,U}(-score_decrement, score, noise,
                                     subtraces, retval, discard, num_nonempty,
                                     Dict{Int,Any}())
    process_all_retained!(gen_fn, args, argdiff, assmt, prev_length, new_length,    
                          retained_and_constrained, state)
    process_all_new!(gen_fn, args, assmt, prev_length, new_length, state)

    # retdiff
    retdiff = vector_compute_retdiff(state.isdiff_retdiffs, new_length, prev_length)

    # new trace
    new_trace = VectorTrace{MapType,T,U}(gen_fn, state.subtraces, state.retval, args,  
        state.score, state.noise, new_length, state.num_nonempty)

    return (new_trace, state.weight, retdiff, discard)
end
