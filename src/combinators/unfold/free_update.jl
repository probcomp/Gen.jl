mutable struct UnfoldFreeUpdateState{T,U}
    init_state::T
    weight::Float64
    score::Float64
    noise::Float64
    subtraces::PersistentVector{U}
    retval::PersistentVector{T}
    num_nonempty::Int
    isdiff_retdiffs::Dict{Int,Any}
end

function process_retained!(gen_fn::Unfold{T,U}, params::Tuple,
                           selection::AddressSet, key::Int, kernel_argdiff,
                           state::UnfoldFreeUpdateState{T,U}) where {T,U}
    local subtrace::U
    local prev_subtrace::U
    local prev_state::T
    local new_state::T

    if has_internal_node(selection, key)
        subselection = get_internal_node(selection, key)
    else
        subselection = EmptyAddressSet()
    end
    prev_state = (key == 1) ? state.init_state : state.retval[key-1]
    kernel_args = (key, prev_state, params...)

    # get new subtrace with recursive call to free_update()
    prev_subtrace = state.subtraces[key]
    (subtrace, weight, subretdiff) = free_update(
        kernel_args, kernel_argdiff, prev_subtrace, subselection)

    # retrieve retdiff
    is_state_diff = !isnodiff(subretdiff)
    if is_state_diff
        state.isdiff_retdiffs[key] = subretdiff
    end

    # update state
    state.weight += weight
    state.score += (get_score(subtrace) - get_score(prev_subtrace))
    state.noise += (project(subtrace, EmptyAddressSet()) - project(subtrace, EmptyAddressSet()))
    state.subtraces = assoc(state.subtraces, key, subtrace)
    new_state = get_retval(subtrace)
    state.retval = assoc(state.retval, key, new_state)
    subtrace_empty = isempty(get_assignment(subtrace))
    prev_subtrace_empty = isempty(get_assignment(prev_subtrace))
    if !subtrace_empty && prev_subtrace_empty
        state.num_nonempty += 1
    elseif subtrace_empty && !prev_subtrace_empty
        state.num_nonempty -= 1
    end

    is_state_diff
end

function process_new!(gen_fn::Unfold{T,U}, params::Tuple, selection::AddressSet, key::Int,
                      state::UnfoldFreeUpdateState{T,U}) where {T,U}
    local subtrace::U
    local prev_state::T
    local new_state::T

    if has_internal_node(selection, key)
        error("Cannot select new addresses in free_update")
    end
    prev_state = (key == 1) ? state.init_state : state.retval[key-1]
    kernel_args = (key, prev_state, params...)

    # get subtrace and weight
    (subtrace, weight) = initialize(gen_fn.kernel, kernel_args, EmptyAssignment())

    # update state
    state.weight += weight
    state.score += get_score(subtrace)
    new_state = get_retval(subtrace)
    @assert key > length(state.subtraces)
    state.subtraces = push(state.subtraces, subtrace)
    state.retval = push(state.retval, new_state)
    @assert length(state.subtraces) == key
    if !isempty(get_assignment(subtrace))
        state.num_nonempty += 1
    end
end

function free_update(args::Tuple, ::NoArgDiff,
                     trace::VectorTrace{UnfoldType,T,U},
                     selection::AddressSet) where {T,U}
    argdiff = UnfoldCustomArgDiff(false, false)
    free_update(args, argdiff, trace, selection)
end

function free_update(args::Tuple, ::UnknownArgDiff,
                     trace::VectorTrace{UnfoldType,T,U},
                     selection::AddressSet) where {T,U}
    argdiff = UnfoldCustomArgDiff(true, true)
    free_update(args, argdiff, trace, selection)
end

function free_update(args::Tuple, argdiff::UnfoldCustomArgDiff,
                     trace::VectorTrace{UnfoldType,T,U},
                     selection::AddressSet) where {T,U}
    gen_fn = trace.gen_fn
    (new_length, init_state, params) = unpack_args(args)
    check_length(new_length)
    prev_args = get_args(trace)
    prev_length = prev_args[1]
    retained_and_selected = get_retained_and_selected(selection, prev_length, new_length)

    # handle removed applications
    (num_nonempty, score_decrement, noise_decrement) = vector_fix_free_update_delete(
        new_length, prev_length, trace)
    (subtraces, retval) = vector_remove_deleted_applications(
        trace.subtraces, trace.retval, prev_length, new_length)
    score = trace.score - score_decrement
    noise = trace.noise - noise_decrement

    # handle retained and new applications
    state = UnfoldFreeUpdateState{T,U}(init_state, -noise_decrement, score, noise,
        subtraces, retval, num_nonempty, Dict{Int,Any}())
    process_all_retained!(gen_fn, params, argdiff, selection, prev_length, new_length,    
                          retained_and_selected, state)
    process_all_new!(gen_fn, params, selection, prev_length, new_length, state)

    # retdiff
    retdiff = vector_compute_retdiff(state.isdiff_retdiffs, new_length, prev_length)

    # new trace
    new_trace = VectorTrace{MapType,T,U}(gen_fn, state.subtraces, state.retval, args,  
        state.score, state.noise, new_length, state.num_nonempty)

    (new_trace, state.weight, retdiff)
end
