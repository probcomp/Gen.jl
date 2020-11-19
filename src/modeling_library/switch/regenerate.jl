mutable struct SwitchRegenerateState{T}
    weight::Float64
    score::Float64
    noise::Float64
    prev_trace::Trace
    trace::Trace
    index::Int
    retdiff::Diff
    SwitchRegenerateState{T}(weight::Float64, score::Float64, noise::Float64, prev_trace::Trace) where T = new{T}(weight, score, noise, prev_trace)
end

function regenerate_recurse_merge(prev_choices::ChoiceMap, selection::Selection)
    prev_choice_submap_iterator = get_submaps_shallow(prev_choices)
    prev_choice_value_iterator = get_values_shallow(prev_choices)
    new_choices = DynamicChoiceMap()
    for (key, value) in prev_choice_value_iterator
        key in selection && continue
        set_value!(new_choices, key, value)
    end
    for (key, node1) in prev_choice_submap_iterator
        if key in selection
            subsel = get_subselection(selection, key)
            node = regenerate_recurse_merge(node1, subsel)
            set_submap!(new_choices, key, node)
        else
            set_submap!(new_choices, key, node1)
        end
    end
    return new_choices
end

function process!(gen_fn::Switch{C, N, K, T},
                  index::Int,
                  index_argdiff::UnknownChange,
                  args::Tuple,
                  kernel_argdiffs::Tuple,
                  selection::Selection, 
                  state::SwitchRegenerateState{T}) where {C, N, K, T}
    branch_fn = getfield(gen_fn.mix, index)
    merged = regenerate_recurse_merge(get_choices(state.prev_trace), selection)
    new_trace, weight = generate(branch_fn, args, merged)
    retdiff = UnknownChange()
    weight -= get_score(state.prev_trace)
    state.index = index
    state.weight = weight
    state.noise = project(new_trace, EmptySelection()) - project(state.prev_trace, EmptySelection())
    state.score = get_score(new_trace)
    state.trace = new_trace
    state.retdiff = retdiff
end

function process!(gen_fn::Switch{C, N, K, T},
                  index::Int,
                  index_argdiff::NoChange,
                  args::Tuple,
                  kernel_argdiffs::Tuple,
                  selection::Selection, 
                  state::SwitchRegenerateState{T}) where {C, N, K, T}
    new_trace, weight, retdiff = regenerate(getfield(state.prev_trace, :branch), args, kernel_argdiffs, selection)
    state.index = index
    state.weight = weight
    state.noise = project(new_trace, EmptySelection()) - project(state.prev_trace, EmptySelection())
    state.score = get_score(new_trace)
    state.trace = new_trace
    state.retdiff = retdiff
end

@inline process!(gen_fn::Switch{C, N, K, T}, index::C, index_argdiff::Diff, args::Tuple, kernel_argdiffs::Tuple, selection::Selection, state::SwitchRegenerateState{T}) where {C, N, K, T} = process!(gen_fn, getindex(gen_fn.cases, index), index_argdiff, args, kernel_argdiffs, selection, state)

function regenerate(trace::SwitchTrace{T},
                    args::Tuple, 
                    argdiffs::Tuple,
                    selection::Selection) where T
    gen_fn = trace.gen_fn
    index, index_argdiff = args[1], argdiffs[1]
    state = SwitchRegenerateState{T}(0.0, 0.0, 0.0, trace)
    process!(gen_fn, index, index_argdiff, args[2 : end], argdiffs[2 : end], selection, state)
    return SwitchTrace(gen_fn, state.index, state.trace, get_retval(state.trace), args, state.score, state.noise), state.weight, state.retdiff
end
