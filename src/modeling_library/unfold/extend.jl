mutable struct UnfoldExtendState{T,U}
    init_state::T
    weight::Float64
    score::Float64
    noise::Float64
    subtraces::PersistentVector{U}
    retval::PersistentVector{T}
    num_nonempty::Int
    updated_retdiffs::Dict{Int,Diff}
end

function process_retained!(gen_fn::Unfold{T,U}, params::Tuple,
                           choices::ChoiceMap, key::Int, kernel_argdiffs::Tuple,
                           state::UnfoldExtendState{T,U}) where {T,U}
    local subtrace::U
    local prev_subtrace::U
    local prev_state::T
    local new_state::T

    submap = get_submap(choices, key)
    prev_state = (key == 1) ? state.init_state : state.retval[key-1]
    kernel_args = (key, prev_state, params...)

    # get new subtrace with recursive call to extend()
    prev_subtrace = state.subtraces[key]
    (subtrace, weight, retdiff) = extend(
        prev_subtrace, kernel_args, kernel_argdiffs, submap)

    # retrieve retdiff
    if retdiff != NoChange()
        state.updated_retdiffs[key] = retdiff
    end

    # update state
    state.weight += weight
    state.score += (get_score(subtrace) - get_score(prev_subtrace))
    state.noise += (project(subtrace, EmptyAddressSet()) - project(subtrace, EmptyAddressSet()))
    state.subtraces = assoc(state.subtraces, key, subtrace)
    new_state = get_retval(subtrace)
    state.retval = assoc(state.retval, key, new_state)
    subtrace_empty = isempty(get_choices(subtrace))
    prev_subtrace_empty = isempty(get_choices(prev_subtrace))
    @assert !(subtrace_empty && !prev_subtrace_empty)
    if !subtrace_empty && prev_subtrace_empty
        state.num_nonempty += 1
    end

    retdiff
end

function process_new!(gen_fn::Unfold{T,U}, params::Tuple, choices, key::Int,
                      state::UnfoldExtendState{T,U}) where {T,U}
    local subtrace::U
    local prev_state::T
    local new_state::T

    submap = get_submap(choices, key)
    prev_state = (key == 1) ? state.init_state : state.retval[key-1]
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

function extend(trace::VectorTrace{UnfoldType,T,U},
                args::Tuple, argdiffs::Tuple,
                choices::ChoiceMap) where {T,U}
    gen_fn = trace.gen_fn
    (new_length, init_state, params) = unpack_args(args)
    check_length(new_length)
    prev_args = get_args(trace)
    prev_length = prev_args[1]
    retained_and_constrained = get_retained_and_constrained(choices, prev_length, new_length)

    # there can be no removed applications
    if new_length < prev_length
        error("Cannot decrease number of applications from $new_length to $prev_length in map extend")
    end

    # handle retained and new applications
    state = UnfoldExtendState{T,U}(init_state, 0., trace.score, trace.noise,
        trace.subtraces, trace.retval, trace.num_nonempty, Dict{Int,Any}())
    process_all_retained!(gen_fn, params, argdiffs, choices, prev_length, new_length,    
                          retained_and_constrained, state)
    process_all_new!(gen_fn, params, choices, prev_length, new_length, state)

    # retdiff
    retdiff = vector_compute_retdiff(state.updated_retdiffs, new_length, prev_length)

    # new trace
    new_trace = VectorTrace{UnfoldType,T,U}(gen_fn, state.subtraces, state.retval, args,  
        state.score, state.noise, new_length, state.num_nonempty)

    (new_trace, state.weight, retdiff)
end
