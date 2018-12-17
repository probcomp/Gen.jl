mutable struct UnfoldFixUpdateState{T,U}
    init_state::T
    weight::Float64
    score::Float64
    noise::Float64
    subtraces::PersistentVector{U}
    retval::PersistentVector{T}
    discard::DynamicAssignment
    num_nonempty::Int
    isdiff_retdiffs::Dict{Int,Any}
end

function process_retained!(gen_fn::Unfold{T,U}, params::Tuple,
                           assmt::Assignment, key::Int, kernel_argdiff,
                           state::UnfoldFixUpdateState{T,U}) where {T,U}
    local subtrace::U
    local prev_subtrace::U
    local prev_state::T
    local new_state::T

    subassmt = get_subassmt(assmt, key)
    prev_state = (key == 1) ? state.init_state : state.retval[key-1]
    kernel_args = (key, prev_state, params...)

    # get new subtrace with recursive call to fix_update()
    prev_subtrace = state.subtraces[key]
    (subtrace, weight, discard, subretdiff) = fix_update(
        kernel_args, kernel_argdiff, prev_subtrace, subassmt)

    # retrieve retdiff
    is_state_diff = !isnodiff(subretdiff)
    if is_state_diff
        state.isdiff_retdiffs[key] = subretdiff
    end

    # update state
    state.weight += weight
    set_subassmt!(state.discard, key, discard)
    state.score += (get_score(subtrace) - get_score(prev_subtrace))
    state.noise += (project(subtrace, EmptyAddressSet()) - project(subtrace, EmptyAddressSet()))
    state.subtraces = assoc(state.subtraces, key, subtrace)
    new_state = get_retval(subtrace)
    state.retval = assoc(state.retval, key, new_state)
    subtrace_empty = isempty(get_assmt(subtrace))
    prev_subtrace_empty = isempty(get_assmt(prev_subtrace))
    if !subtrace_empty && prev_subtrace_empty
        state.num_nonempty += 1
    elseif subtrace_empty && !prev_subtrace_empty
        state.num_nonempty -= 1
    end

    is_state_diff
end

function process_new!(gen_fn::Unfold{T,U}, params::Tuple, assmt, key::Int,
                      state::UnfoldFixUpdateState{T,U}) where {T,U}
    local subtrace::U
    local prev_state::T
    local new_state::T

    if !isempty(get_subassmt(assmt, key))
        error("Cannot constrain new addresses in fix_update")
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
    if !isempty(get_assmt(subtrace))
        state.num_nonempty += 1
    end
end

function fix_update(args::Tuple, ::NoArgDiff,
                      trace::VectorTrace{UnfoldType,T,U},
                      assmt::Assignment) where {T,U}
    argdiff = UnfoldCustomArgDiff(false, false)
    fix_update(args, argdiff, trace, assmt)
end

function fix_update(args::Tuple, ::UnknownArgDiff,
                      trace::VectorTrace{UnfoldType,T,U},
                      assmt::Assignment) where {T,U}
    argdiff = UnfoldCustomArgDiff(true, true)
    fix_update(args, argdiff, trace, assmt)
end

function fix_update(args::Tuple, argdiff::UnfoldCustomArgDiff,
                      trace::VectorTrace{UnfoldType,T,U},
                      assmt::Assignment) where {T,U}
    gen_fn = trace.gen_fn
    (new_length, init_state, params) = unpack_args(args)
    check_length(new_length)
    prev_args = get_args(trace)
    prev_length = prev_args[1]
    retained_and_constrained = get_retained_and_constrained(assmt, prev_length, new_length)

    # handle removed applications
    (num_nonempty, score_decrement, noise_decrement) = vector_fix_free_update_delete(
        new_length, prev_length, trace)
    (subtraces, retval) = vector_remove_deleted_applications(
        trace.subtraces, trace.retval, prev_length, new_length)
    score = trace.score - score_decrement
    noise = trace.noise - noise_decrement

    # handle retained and new applications
    discard = DynamicAssignment()
    state = UnfoldFixUpdateState{T,U}(init_state, -noise_decrement, score, noise,
        subtraces, retval, discard, num_nonempty, Dict{Int,Any}())
    process_all_retained!(gen_fn, params, argdiff, assmt, prev_length, new_length,    
                          retained_and_constrained, state)
    process_all_new!(gen_fn, params, assmt, prev_length, new_length, state)

    # retdiff
    retdiff = vector_compute_retdiff(state.isdiff_retdiffs, new_length, prev_length)

    # new trace
    new_trace = VectorTrace{MapType,T,U}(gen_fn, state.subtraces, state.retval, args,  
        state.score, state.noise, new_length, state.num_nonempty)

    (new_trace, state.weight, discard, retdiff)
end
