mutable struct UnfoldExtendUpdateState{T,U}
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
                           assmt::Assignment, key::Int, kernel_argdiff,
                           state::UnfoldExtendUpdateState{T,U}) where {T,U}
    local subtrace::U
    local prev_subtrace::U
    local prev_state::T
    local new_state::T

    subassmt = get_subassmt(assmt, key)
    prev_state = (key == 1) ? state.init_state : state.retval[key-1]
    kernel_args = (key, prev_state, params...)

    # get new subtrace with recursive call to extend()
    prev_subtrace = state.subtraces[key]
    (subtrace, weight, subretdiff) = extend(
        kernel_args, kernel_argdiff, prev_subtrace, subassmt)

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
    subtrace_empty = isempty(get_assmt(subtrace))
    prev_subtrace_empty = isempty(get_assmt(prev_subtrace))
    @assert !(subtrace_empty && !prev_subtrace_empty)
    if !subtrace_empty && prev_subtrace_empty
        state.num_nonempty += 1
    end

    is_state_diff
end

function process_new!(gen_fn::Unfold{T,U}, params::Tuple, assmt, key::Int,
                      state::UnfoldExtendUpdateState{T,U}) where {T,U}
    local subtrace::U
    local prev_state::T
    local new_state::T

    subassmt = get_subassmt(assmt, key)
    prev_state = (key == 1) ? state.init_state : state.retval[key-1]
    kernel_args = (key, prev_state, params...)

    # get subtrace and weight
    (subtrace, weight) = initialize(gen_fn.kernel, kernel_args, subassmt)

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

function extend(args::Tuple, ::NoArgDiff,
                trace::VectorTrace{UnfoldType,T,U},
                assmt::Assignment) where {T,U}
    argdiff = UnfoldCustomArgDiff(false, false)
    extend(args, argdiff, trace, assmt)
end

function extend(args::Tuple, ::UnknownArgDiff,
                trace::VectorTrace{UnfoldType,T,U},
               assmt::Assignment) where {T,U}
    argdiff = UnfoldCustomArgDiff(true, true)
    extend(args, argdiff, trace, assmt)
end

function extend(args::Tuple, argdiff::UnfoldCustomArgDiff,
                      trace::VectorTrace{UnfoldType,T,U},
                      assmt::Assignment) where {T,U}
    gen_fn = trace.gen_fn
    (new_length, init_state, params) = unpack_args(args)
    check_length(new_length)
    prev_args = get_args(trace)
    prev_length = prev_args[1]
    retained_and_constrained = get_retained_and_constrained(assmt, prev_length, new_length)

    # there can be no removed applications
    if new_length < prev_length
        error("Cannot decrease number of applications from $new_length to $prev_length in map extend")
    end

    # handle retained and new applications
    state = UnfoldExtendUpdateState{T,U}(init_state, 0., trace.score, trace.noise,
        trace.subtraces, trace.retval, trace.num_nonempty, Dict{Int,Any}())
    process_all_retained!(gen_fn, params, argdiff, assmt, prev_length, new_length,    
                          retained_and_constrained, state)
    process_all_new!(gen_fn, params, assmt, prev_length, new_length, state)

    # retdiff
    retdiff = vector_compute_retdiff(state.isdiff_retdiffs, new_length, prev_length)

    # new trace
    new_trace = VectorTrace{UnfoldType,T,U}(gen_fn, state.subtraces, state.retval, args,  
        state.score, state.noise, new_length, state.num_nonempty)

    (new_trace, state.weight, retdiff)
end
