mutable struct SwitchUpdateState{T,U}
    weight::Float64
    score::Float64
    noise::Float64
    cond::Bool
    subtrace::U
    retval::T
    discard::DynamicChoiceMap
    updated_retdiff::Diff
end

function process!(gen_fn::Switch{T1, T2, Tr}, 
                           branch_p::Float64,
                           args::Tuple,
                           choices::ChoiceMap, 
                           kernel_argdiffs::Tuple,
                           state::SwitchUpdateState{Union{T1, T2}, Tr}) where {T1, T2, Tr}
    local subtrace::Tr
    local prev_subtrace::Tr
    local retval::T

    # get new subtrace with recursive call to update()
    submap = get_submap(choices, :branch)
    prev_subtrace = state.subtrace
    (subtrace, weight, retdiff, discard) = update(prev_subtrace, kernel_args, kernel_argdiffs, submap)

    # retrieve retdiff
    if retdiff != NoChange()
        state.updated_retdiff = retdiff
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

function update(trace::Switch{T1, T2, Tr}, 
                args::Tuple, 
                argdiffs::Tuple,
                choices::ChoiceMap) where {T1, T2, Tr}
    gen_fn = trace.gen_fn
    branch_p = args[1]
    return (new_trace, state.weight, retdiff, discard)
end
