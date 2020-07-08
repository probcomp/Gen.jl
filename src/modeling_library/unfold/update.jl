mutable struct UnfoldUpdateState{T,U}
    init_state::T
    weight::Float64
    score::Float64
    noise::Float64
    subtraces::PersistentVector{U}
    retval::PersistentVector{T}
    discard::DynamicChoiceMap
    num_nonempty::Int
    updated_retdiffs::Dict{Int,Diff}
end

function process_all_retained!(gen_fn::Unfold{T,U}, params::Tuple, argdiffs::Tuple,
    spec, prev_length::Int, new_length::Int,
    retained_and_targeted::Set{Int}, externally_constrained_addrs, state) where {T,U}

    len_diff = argdiffs[1]
    init_state_diff = argdiffs[2]
    param_diffs = argdiffs[3:end] # a tuple of diffs

    if any(diff != NoChange() for diff in param_diffs)

        # visit every retained kernel application
        state_diff = init_state_diff
        for key = 1:min(prev_length, new_length)
            state_diff = process_retained!(gen_fn, params, spec,
                key, (NoChange(), state_diff, param_diffs...), externally_constrained_addrs, state)
        end

    else
        # every parameter diff is NoChange()

        # visit only certain retained kernel applications
        to_visit::Vector{Int} = sort(collect(retained_and_targeted))
        key = 0
        state_diff = init_state_diff
        if state_diff != NoChange()
            key = 1
            visit = true
            while visit && key <= min(prev_length, new_length)
                state_diff = process_retained!(gen_fn, params, spec,
                        key, (NoChange(), state_diff, param_diffs...), externally_constrained_addrs, state)
                key += 1
                visit = (state_diff != NoChange())
            end
        end
        for i = 1:length(to_visit)
            if key > to_visit[i]
                # we have already visited it
                continue
            end
            key = to_visit[i]
            visit = true
            while visit && key <= min(prev_length, new_length)
                state_diff = process_retained!(gen_fn, params, spec,
                        key, (NoChange(), state_diff, param_diffs...), externally_constrained_addrs, state)
                key += 1
                visit = (state_diff != NoChange())
            end
        end
    end
end

"""
Process all new applications.
"""
function process_all_new!(gen_fn::Unfold{T,U}, params::Tuple, choices,
    prev_len::Int, new_len::Int, state) where {T,U}
    for key = prev_len + 1:new_len
        process_new!(gen_fn, params, choices, key, state)
    end
end

function process_retained!(gen_fn::Unfold{T,U}, params::Tuple,
                           spec::UpdateSpec, key::Int, kernel_argdiffs::Tuple,
                           externally_constrained_addrs::Selection, state::UnfoldUpdateState{T,U}) where {T,U}
    local subtrace::U
    local prev_subtrace::U
    local prev_state::T
    local new_state::T

    subspec = get_subtree(spec, key)
    sub_ext_const_addrs = get_subtree(externally_constrained_addrs, key)
    prev_state = (key == 1) ? state.init_state : state.retval[key - 1]
    kernel_args = (key, prev_state, params...)

    # get new subtrace with recursive call to update()
    prev_subtrace = state.subtraces[key]
    (subtrace, weight, retdiff, discard) = update(
        prev_subtrace, kernel_args, kernel_argdiffs, subspec, sub_ext_const_addrs)

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
    new_state = get_retval(subtrace)
    state.retval = assoc(state.retval, key, new_state)
    subtrace_empty = isempty(get_choices(subtrace))
    prev_subtrace_empty = isempty(get_choices(prev_subtrace))
    if !subtrace_empty && prev_subtrace_empty
        state.num_nonempty += 1
    elseif subtrace_empty && !prev_subtrace_empty
        state.num_nonempty -= 1
    end

    retdiff
end

function process_new!(gen_fn::Unfold{T,U}, params::Tuple, spec, key::Int,
                      state::UnfoldUpdateState{T,U}) where {T,U}
    local subtrace::U
    local prev_state::T
    local new_state::T

    submap = get_subtree(spec, key)
    prev_state = (key == 1) ? state.init_state : state.retval[key - 1]
    kernel_args = (key, prev_state, params...)

    # get subtrace and weight
    (subtrace, weight) = generate(gen_fn.kernel, kernel_args, submap)

    # update state
    state.weight += weight
    state.score += get_score(subtrace)
    new_state = get_retval(subtrace)
    @assert key > length(state.subtraces)
    state.subtraces = push(state.subtraces, subtrace)
    state.retval = push(state.retval, new_state)
    @assert length(state.subtraces) == key
    if !isempty(get_choices(subtrace))
        state.num_nonempty += 1
    end
end

function update(trace::VectorTrace{UnfoldType,T,U},
                args::Tuple, argdiffs::Tuple,
                spec::UpdateSpec, externally_constrained_addrs::Selection) where {T,U}
    gen_fn = trace.gen_fn
    (new_length, init_state, params) = unpack_args(args)
    check_length(new_length)
    prev_args = get_args(trace)
    prev_length = prev_args[1]
    retained_and_specd = get_retained_and_specd(spec, prev_length, new_length)

    # handle removed applications
    (discard, num_nonempty, score_decrement, noise_decrement, weight_decrement) = vector_update_delete(
        new_length, prev_length, trace, externally_constrained_addrs)
    (subtraces, retval) = vector_remove_deleted_applications(
        trace.subtraces, trace.retval, prev_length, new_length)
    score = trace.score - score_decrement
    noise = trace.noise - noise_decrement

    # handle retained and new applications
    state = UnfoldUpdateState{T,U}(init_state, -weight_decrement, score, noise,
        subtraces, retval, discard, num_nonempty, Dict{Int,Diff}())
    process_all_retained!(gen_fn, params, argdiffs, spec, prev_length, new_length,    
                        retained_and_specd, externally_constrained_addrs,  state)
    process_all_new!(gen_fn, params, spec, prev_length, new_length, state)

    # retdiff
    retdiff = vector_compute_retdiff(state.updated_retdiffs, new_length, prev_length)

    # new trace
    new_trace = VectorTrace{UnfoldType,T,U}(gen_fn, state.subtraces, state.retval, args,  
        state.score, state.noise, new_length, state.num_nonempty)

    (new_trace, state.weight, retdiff, discard)
end
