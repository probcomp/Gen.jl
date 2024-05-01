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

    # Add (address, value) to new_choices from prev_choices if address does not occur in choices.
    for (address, value) in prev_choice_value_iterator
        address in keys(choice_value_iterator) && continue
        set_value!(new_choices, address, value)
    end

    # Add (address, submap) to new_choices from prev_choices if address does not occur in choices.
    # If it does, enter a recursive call to update_recurse_merge.
    for (address, node1) in prev_choice_submap_iterator
        if address in keys(choice_submap_iterator)
            node2 = get_submap(choices, address)
            node = update_recurse_merge(node1, node2)
            set_submap!(new_choices, address, node)
        else
            set_submap!(new_choices, address, node1)
        end
    end

    # Add (address, value) from choices to new_choices. This is okay because we've excluded any conflicting addresses from the prev_choices above.
    for (address, value) in choice_value_iterator
        set_value!(new_choices, address, value)
    end

    sel, _ = zip(prev_choice_submap_iterator...)
    comp = complement(select(sel...))
    for (address, node) in get_submaps_shallow(get_selected(choices, comp))
        set_submap!(new_choices, address, node)
    end
    return new_choices
end

@doc(
"""
update_recurse_merge(prev_choices::ChoiceMap, choices::ChoiceMap)

Returns choices that are in constraints, merged with all choices in the previous trace that do not have the same address as some choice in the constraints."
""", update_recurse_merge)

function update_discard(prev_choices::ChoiceMap, choices::ChoiceMap, new_choices::ChoiceMap)
    discard = choicemap()
    for (k, v) in get_submaps_shallow(prev_choices)
        new_submap = get_submap(new_choices, k)
        choices_submap = get_submap(choices, k)
        sub_discard = update_discard(v, choices_submap, new_submap)
        set_submap!(discard, k, sub_discard)
    end
    for (k, v) in get_values_shallow(prev_choices)
        if (!has_value(new_choices, k) || has_value(choices, k))
            set_value!(discard, k, v)
        end
    end
    discard
end

@doc(
"""
update_discard(prev_choices::ChoiceMap, choices::ChoiceMap, new_choices::ChoiceMap)

Returns choices from previous trace that:
   1. have an address which does not appear in the new trace.
   2. have an address which does appear in the constraints.
""", update_discard)

@inline update_discard(prev_trace::Trace, choices::ChoiceMap, new_trace::Trace) = update_discard(get_choices(prev_trace), choices, get_choices(new_trace))

function process!(
    rng::AbstractRNG,
    gen_fn::Switch{C, N, K, T},
    index::Int,
    index_argdiff::UnknownChange,
    args::Tuple,
    kernel_argdiffs::Tuple,
    choices::ChoiceMap,
    state::SwitchUpdateState{T}
) where {C, N, K, T}

    # Generate new trace.
    merged = update_recurse_merge(get_choices(state.prev_trace), choices)
    branch_fn = getfield(gen_fn.branches, index)
    new_trace, weight = generate(rng, branch_fn, args, merged)
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

function process!(
    rng::AbstractRNG,
    gen_fn::Switch{C, N, K, T},
    index::Int,
    index_argdiff::NoChange, # TODO: Diffed wrapper?
    args::Tuple,
    kernel_argdiffs::Tuple,
    choices::ChoiceMap,
    state::SwitchUpdateState{T}
) where {C, N, K, T}

    # Update trace.
    new_trace, weight, retdiff, discard = update(rng, getfield(state.prev_trace, :branch), args, kernel_argdiffs, choices)

    # Set state.
    state.index = index
    state.weight = weight
    state.noise = project(new_trace, EmptySelection()) - project(state.prev_trace, EmptySelection())
    state.score = get_score(new_trace)
    state.trace = new_trace
    state.updated_retdiff = retdiff
    state.discard = discard
end

@inline process!(rng::AbstractRNG, gen_fn::Switch{C, N, K, T}, index::C, index_argdiff::Diff, args::Tuple, kernel_argdiffs::Tuple, choices::ChoiceMap, state::SwitchUpdateState{T}) where {C, N, K, T} = process!(rng, gen_fn, getindex(gen_fn.cases, index), index_argdiff, args, kernel_argdiffs, choices, state)

function update(
    rng::AbstractRNG,
    trace::SwitchTrace{A, T, U},
    args::Tuple,
    argdiffs::Tuple,
    choices::ChoiceMap
) where {A, T, U}
    gen_fn = trace.gen_fn
    index, index_argdiff = args[1], argdiffs[1]
    state = SwitchUpdateState{T}(0.0, 0.0, 0.0, trace)
    process!(rng, gen_fn, index, index_argdiff,
             args[2 : end], argdiffs[2 : end], choices, state)
    return SwitchTrace(gen_fn, state.trace,
                       get_retval(state.trace), args,
                       state.score, state.noise), state.weight, state.updated_retdiff, state.discard
end
