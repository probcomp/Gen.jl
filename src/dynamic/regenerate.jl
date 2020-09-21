mutable struct GFRegenerateState
    prev_trace::DynamicDSLTrace
    trace::DynamicDSLTrace
    selection::Selection
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
end

function GFRegenerateState(gen_fn, args, prev_trace,
                           selection, params)
    visitor = AddressVisitor()
    GFRegenerateState(prev_trace, DynamicDSLTrace(gen_fn, args), selection,
        0., visitor, params)
end

function traceat(state::GFRegenerateState, dist::Distribution{T},
                 args, key) where {T}
    local prev_retval::T
    local retval::T

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for previous choice at this key
    has_previous = has_choice(state.prev_trace, key)
    if has_previous
        prev_choice = get_choice(state.prev_trace, key)
        prev_retval = prev_choice.retval
        prev_score = prev_choice.score
    end

    # check whether the key was selected
    in_selection = key in state.selection

    # get return value
    if has_previous && in_selection
        retval = random(dist, args...)
    elseif has_previous
        retval = prev_retval
    else
        retval = random(dist, args...)
    end

    # compute logpdf
    score = logpdf(dist, retval, args...)

    # update weight
    if has_previous && !in_selection
        state.weight += score - prev_score
    end

    # add to the trace
    add_choice!(state.trace, key, retval, score)

    retval
end

function traceat(state::GFRegenerateState, gen_fn::GenerativeFunction{T,U},
                 args, key) where {T,U}
    local prev_retval::T
    local trace::U
    local retval::T

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check whether the key was selected
    subselection = state.selection[key]

    # get subtrace
    has_previous = has_call(state.prev_trace, key)
    if has_previous
        prev_call = get_call(state.prev_trace, key)
        prev_subtrace = prev_call.subtrace
        get_gen_fn(prev_subtrace) === gen_fn || gen_fn_changed_error(key)
        (subtrace, weight, _) = regenerate(
            prev_subtrace, args, map((_) -> UnknownChange(), args), subselection)
    else
        (subtrace, weight) = generate(gen_fn, args, EmptyChoiceMap())
    end

    # update weight
    state.weight += weight

    # add to the trace
    add_call!(state.trace, key, subtrace)

    # get return value
    retval = get_retval(subtrace)

    retval
end

function splice(state::GFRegenerateState, gen_fn::DynamicDSLFunction,
                args::Tuple)
    prev_params = state.params
    state.params = gen_fn.params
    retval = exec(gen_fn, state, args)
    state.params = prev_params
    retval
end

function regenerate_delete_recurse(prev_trie::Trie{Any,ChoiceOrCallRecord},
                             visited::EmptySelection)
    noise = 0.
    for (key, choice_or_call) in get_leaf_nodes(prev_trie)
        if !choice_or_call.is_choice
            noise += choice_or_call.noise
        end
    end
    for (key, subtrie) in get_internal_nodes(prev_trie)
        noise += regenerate_delete_recurse(subtrie, EmptySelection())
    end
    noise
end

function regenerate_delete_recurse(prev_trie::Trie{Any,ChoiceOrCallRecord},
                             visited::DynamicSelection)
    noise = 0.
    for (key, choice_or_call) in get_leaf_nodes(prev_trie)
        if !(key in visited) && !choice_or_call.is_choice
            noise += choice_or_call.noise
        end
    end
    for (key, subtrie) in get_internal_nodes(prev_trie)
        subvisited = visited[key]
        noise += regenerate_delete_recurse(subtrie, subvisited)
    end
    noise
end

function regenerate(trace::DynamicDSLTrace, args::Tuple, argdiffs::Tuple,
                    selection::Selection)
    gen_fn = trace.gen_fn
    state = GFRegenerateState(gen_fn, args, trace, selection, gen_fn.params)
    retval = exec(gen_fn, state, args)
    set_retval!(state.trace, retval)
    visited = state.visitor.visited
    state.weight -= regenerate_delete_recurse(trace.trie, visited)
    (state.trace, state.weight, UnknownChange())
end
