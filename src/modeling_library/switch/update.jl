mutable struct SwitchUpdateState{T}
    weight::Float64
    score::Float64
    noise::Float64
    prev_trace::Trace
    trace::Trace
    index::Int
    discard::ChoiceMap
    updated_retdiff::Diff
    SwitchUpdateState{T}(weight::Float64, score::Float64, noise::Float64, prev_trace::Trace) where T = new{T}(weight, score, noise, prev_trace)
end

function update_recurse_merge(prev_choices::ChoiceMap, choices::ChoiceMap)
    prev_choice_submap_iterator = get_submaps_shallow(prev_choices)
    prev_choice_value_iterator = get_values_shallow(prev_choices)
    choice_submap_iterator = get_submaps_shallow(choices)
    choice_value_iterator = get_values_shallow(choices)
    new_choices = DynamicChoiceMap()
    for (key, value) in prev_choice_value_iterator
        key in keys(choice_value_iterator) && continue
        set_value!(new_choices, key, value)
    end
    for (key, node1) in prev_choice_submap_iterator
        if key in keys(choice_submap_iterator)
            node2 = get_submap(choices, key)
            node = update_recurse_merge(node1, node2)
            set_submap!(new_choices, key, node)
        else
            set_submap!(new_choices, key, node1)
        end
    end
    for (key, value) in choice_value_iterator
        set_value!(new_choices, key, value)
    end
    sel, _ = zip(prev_choice_submap_iterator...)
    comp = complement(select(sel...))
    for (key, node) in get_submaps_shallow(get_selected(choices, comp))
        set_submap!(new_choices, key, node)
    end
    return new_choices
end

function update_discard(prev_trace::Trace, choices::ChoiceMap, new_trace::Trace)
    discard = choicemap()
    prev_choices = get_choices(prev_trace)
    for (k, v) in get_submaps_shallow(prev_choices)
        isempty(get_submap(get_choices(new_trace), k)) && continue
        isempty(get_submap(choices, k)) && continue
        set_submap!(discard, k, v)
    end
    for (k, v) in get_values_shallow(prev_choices)
        has_value(get_choices(new_trace), k) || continue
        has_value(choices, k) || continue
        set_value!(discard, k, v)
    end
    discard
end

function process!(gen_fn::Switch{C, N, K, T},
                  index::Int,
                  index_argdiff::UnknownChange, # TODO: Diffed wrapper?
                  args::Tuple,
                  kernel_argdiffs::Tuple,
                  choices::ChoiceMap, 
                  state::SwitchUpdateState{T}) where {C, N, K, T, DV}

    # Generate new trace.
    merged = update_recurse_merge(get_choices(state.prev_trace), choices)
    branch_fn = getfield(gen_fn.mix, index)
    new_trace, weight = generate(branch_fn, args, merged)
    weight -= get_score(state.prev_trace)
    state.discard = update_discard(state.prev_trace, choices, new_trace)

    # Set state.
    state.index = index
    state.weight = weight
    state.noise = project(new_trace, EmptySelection()) - project(state.prev_trace, EmptySelection())
    state.score = get_score(new_trace)
    state.trace = new_trace
    state.updated_retdiff = UnknownChange()
end

function process!(gen_fn::Switch{C, N, K, T},
                  index::Int,
                  index_argdiff::NoChange, # TODO: Diffed wrapper?
                  args::Tuple,
                  kernel_argdiffs::Tuple,
                  choices::ChoiceMap, 
                  state::SwitchUpdateState{T}) where {C, N, K, T}

    # Update trace.
    new_trace, weight, retdiff, discard = update(getfield(state.prev_trace, :branch), args, kernel_argdiffs, choices)

    # Set state.
    state.index = index
    state.weight = weight
    state.noise = project(new_trace, EmptySelection()) - project(state.prev_trace, EmptySelection())
    state.score = get_score(new_trace)
    state.trace = new_trace
    state.updated_retdiff = retdiff
    state.discard = discard
end

@inline process!(gen_fn::Switch{C, N, K, T}, index::C, index_argdiff::Diff, args::Tuple, kernel_argdiffs::Tuple, choices::ChoiceMap, state::SwitchUpdateState{T}) where {C, N, K, T} = process!(gen_fn, getindex(gen_fn.cases, index), index_argdiff, args, kernel_argdiffs, choices, state)

function update(trace::SwitchTrace{T},
                args::Tuple, 
                argdiffs::Tuple,
                choices::ChoiceMap) where T
    gen_fn = trace.gen_fn
    index, index_argdiff = args[1], argdiffs[1]
    state = SwitchUpdateState{T}(0.0, 0.0, 0.0, trace)
    process!(gen_fn, index, index_argdiff, args[2 : end], argdiffs[2 : end], choices, state)
    return SwitchTrace(gen_fn, state.index, state.trace, get_retval(state.trace), args, state.score, state.noise), state.weight, state.updated_retdiff, state.discard
end
