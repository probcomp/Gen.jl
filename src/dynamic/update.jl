mutable struct GFUpdateState
    prev_trace::DynamicDSLTrace
    trace::DynamicDSLTrace
    constraints::Any
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
    discard::DynamicChoiceMap
end

function GFUpdateState(gen_fn, args, prev_trace, constraints, params)
    visitor = AddressVisitor()
    discard = choicemap()
    trace = DynamicDSLTrace(gen_fn, args)
    GFUpdateState(prev_trace, trace, constraints,
        0., visitor, params, discard)
end

function traceat(state::GFUpdateState, gen_fn::GenerativeFunction{T,U},
                 args::Tuple, key) where {T,U}

    local prev_subtrace::U
    local subtrace::U
    local retval::T

    # check key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for constraints at this key
    constraints = get_submap(state.constraints, key)

    # get subtrace
    has_previous = has_call(state.prev_trace, key)
    if has_previous
        prev_call = get_call(state.prev_trace, key)
        prev_subtrace = prev_call.subtrace
        get_gen_fn(prev_subtrace) == gen_fn || gen_fn_changed_error(key)
        (subtrace, weight, _, discard) = update(prev_subtrace,
            args, map((_) -> UnknownChange(), args), constraints)
    else
        (subtrace, weight) = generate(gen_fn, args, constraints)
    end

    # update the weight
    state.weight += weight

    # update discard
    if has_previous
        set_submap!(state.discard, key, discard)
    end

    # add to the trace
    add_call!(state.trace, key, subtrace)

    # get return value
    retval = get_retval(subtrace)

    retval
end

function splice(state::GFUpdateState, gen_fn::DynamicDSLFunction,
                args::Tuple)
    prev_params = state.params
    state.params = gen_fn.params
    retval = exec(gen_fn, state, args)
    state.params = prev_params
    retval
end

function update_delete_recurse(prev_trie::Trie{Any,CallRecord},
                               visited::EmptySelection)
    score = 0.
    for (key, call) in get_leaf_nodes(prev_trie)
        score += call.score
    end
    for (key, subtrie) in get_internal_nodes(prev_trie)
        score += update_delete_recurse(subtrie, EmptySelection())
    end
    score
end

function update_delete_recurse(prev_trie::Trie{Any,CallRecord},
                               visited::DynamicSelection)
    score = 0.
    for (key, call) in get_leaf_nodes(prev_trie)
        if !(key in visited)
            score += call.score
        end
    end
    for (key, subtrie) in get_internal_nodes(prev_trie)
        subvisited = visited[key]
        score += update_delete_recurse(subtrie, subvisited)
    end
    score
end

function update_delete_recurse(prev_trie::Trie{Any,ChoiceOrCallRecord},
                               visited::AllSelection)
    0.
end

function add_unvisited_to_discard!(discard::DynamicChoiceMap,
                                   visited::DynamicSelection,
                                   prev_choices::ChoiceMap)
    for (key, submap) in get_submaps_shallow(prev_choices)
        # if key IS in visited, 
        # the recursive call to update already handled the discard
        # for this entire submap; else we need to handle it
        if !(key in visited)
            @assert isempty(get_submap(discard, key))
            subvisited = visited[key]
            if isempty(subvisited)
                # none of this submap was visited, so we discard the whole thing
                set_submap!(discard, key, submap)
            else
                subdiscard = get_submap(discard, key)
                subdiscard = isempty(subdiscard) ? choicemap() : subdiscard
                add_unvisited_to_discard!(subdiscard, subvisited, submap)
                set_submap!(discard, key, subdiscard)
            end 
        end
    end
end

function update(trace::DynamicDSLTrace, arg_values::Tuple, arg_diffs::Tuple,
                constraints::ChoiceMap)
    gen_fn = trace.gen_fn
    state = GFUpdateState(gen_fn, arg_values, trace, constraints, gen_fn.params)
    retval = exec(gen_fn, state, arg_values)
    set_retval!(state.trace, retval)
    visited = get_visited(state.visitor)
    state.weight -= update_delete_recurse(trace.trie, visited)
    add_unvisited_to_discard!(state.discard, visited, get_choices(trace))
    if !all_visited(visited, constraints)
        error("Did not visit all constraints")
    end
    (state.trace, state.weight, UnknownChange(), state.discard)
end
