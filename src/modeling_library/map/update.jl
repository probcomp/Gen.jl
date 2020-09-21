mutable struct MapUpdateState{T,U}
    weight::Float64
    score::Float64
    noise::Float64
    subtraces::PersistentVector{U}
    retval::PersistentVector{T}
    discard::DynamicChoiceMap
    num_nonempty::Int
    updated_retdiffs::Dict{Int,Diff}
end

function process_retained!(gen_fn::Map{T,U}, args::Tuple,
                           choices::ChoiceMap, key::Int, kernel_argdiffs::Tuple,
                           state::MapUpdateState{T,U}) where {T,U}
    local subtrace::U
    local prev_subtrace::U
    local retval::T

    submap = get_submap(choices, key)
    kernel_args = get_args_for_key(args, key)

    # get new subtrace with recursive call to update()
    prev_subtrace = state.subtraces[key]
    (subtrace, weight, retdiff, discard) = update(
        prev_subtrace, kernel_args, kernel_argdiffs, submap)

    # retrieve retdiff
    if retdiff != NoChange()
        state.updated_retdiffs[key] = retdiff
    end

    # update state
    state.weight += weight
    set_submap!(state.discard, key, discard)
    state.score += (get_score(subtrace) - get_score(prev_subtrace))
    state.noise += (project(subtrace, EmptySelection()) - project(prev_subtrace, EmptySelection()))
    state.subtraces = assoc(state.subtraces, key, subtrace)
    retval = get_retval(subtrace)
    state.retval = assoc(state.retval, key, retval)
    subtrace_empty = isempty(get_choices(subtrace))
    prev_subtrace_empty = isempty(get_choices(prev_subtrace))
    if !subtrace_empty && prev_subtrace_empty
        state.num_nonempty += 1
    elseif subtrace_empty && !prev_subtrace_empty
        state.num_nonempty -= 1
    end
end

function process_new!(gen_fn::Map{T,U}, args::Tuple, choices, key::Int,
                      state::MapUpdateState{T,U}) where {T,U}
    local subtrace::U
    local retval::T

    submap = get_submap(choices, key)
    kernel_args = get_args_for_key(args, key)

    # get subtrace and weight
    (subtrace, weight) = generate(gen_fn.kernel, kernel_args, submap)

    # update state
    state.weight += weight
    state.score += get_score(subtrace)
    retval = get_retval(subtrace)
    @assert key > length(state.subtraces)
    state.subtraces = push(state.subtraces, subtrace)
    state.retval = push(state.retval, retval)
    @assert length(state.subtraces) == key
    if !isempty(get_choices(subtrace))
        state.num_nonempty += 1
    end
end


function update(trace::VectorTrace{MapType,T,U}, args::Tuple, argdiffs::Tuple,
                choices::ChoiceMap) where {T,U}
    gen_fn = trace.gen_fn
    (new_length, prev_length) = get_prev_and_new_lengths(args, trace)
    retained_and_constrained = get_retained_and_constrained(choices, prev_length, new_length)

    # handle removed applications
    (discard, num_nonempty, score_decrement, noise_decrement) = vector_update_delete(
        new_length, prev_length, trace)
    (subtraces, retval) = vector_remove_deleted_applications(
        trace.subtraces, trace.retval, prev_length, new_length)
    score = trace.score - score_decrement
    noise = trace.noise - noise_decrement

    # handle retained and new applications
    state = MapUpdateState{T,U}(-score_decrement, score, noise,
                                     subtraces, retval, discard, num_nonempty,
                                     Dict{Int,Diff}())
    process_all_retained!(gen_fn, args, argdiffs, choices, prev_length, new_length,
                          retained_and_constrained, state)
    process_all_new!(gen_fn, args, choices, prev_length, new_length, state)

    # retdiff
    retdiff = vector_compute_retdiff(state.updated_retdiffs, new_length, prev_length)

    # new trace
    new_trace = VectorTrace{MapType,T,U}(gen_fn, state.subtraces, state.retval, args,
        state.score, state.noise, new_length, state.num_nonempty)

    return (new_trace, state.weight, retdiff, discard)
end
