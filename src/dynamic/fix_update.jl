mutable struct GFFixUpdateState
    prev_trace::DynamicDSLTrace
    trace::DynamicDSLTrace
    constraints::Any
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
    discard::DynamicAssignment
    argdiff::Any
    retdiff::Any
    choicediffs::Trie{Any,Any}
    calldiffs::Trie{Any,Any}
end

function GFFixUpdateState(gen_fn, args, argdiff, prev_trace,
                          constraints, params)
    visitor = AddressVisitor()
    discard = DynamicAssignment()
    GFFixUpdateState(prev_trace, DynamicDSLTrace(gen_fn, args), constraints,
        0., visitor, params, discard, argdiff, DefaultRetDiff(),
        Trie{Any,Any}(), Trie{Any,Any}())
end

function addr(state::GFFixUpdateState, dist::Distribution{T},
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

    # check for constraints at this key
    constrained = has_value(state.constraints, key)
    !constrained && check_no_subassmt(state.constraints, key)
    if constrained && !has_previous
        error("fix_update attempted to constrain a new key: $key")
    end

    # record the previous value as discarded if it is replaced
    if constrained && has_previous
        set_value!(state.discard, key, prev_retval)
    end

    # get return value
    if constrained
        retval = get_value(state.constraints, key)
    elseif has_previous
        retval = prev_retval
    else
        retval = random(dist, args...)
    end

    # compute logpdf
    score = logpdf(dist, retval, args...)

    # update choicediffs
    if has_previous
        set_choice_diff!(state, key, constrained, prev_retval) 
    else
        set_choice_diff_no_prev!(state, key)
    end

    # update weight
    if has_previous
        state.weight += score - prev_score
    end

    # add to the trace
    add_choice!(state.trace, key, ChoiceRecord(retval, score))

    retval 
end

function addr(state::GFFixUpdateState, gen_fn::GenerativeFunction, args, key)
    addr(state, gen_fn, args, key, UnknownArgDiff())
end

function addr(state::GFFixUpdateState, gen_fn::GenerativeFunction{T,U},
              args, key, argdiff) where {T,U}
    local prev_trace::U
    local trace::U
    local retval::T

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for constraints at this key 
    constraints = get_subassmt(state.constraints, key)

    # get subtrace
    has_previous = has_call(state.prev_trace, key)
    if has_previous
        prev_call = get_call(state.prev_trace, key)
        prev_subtrace = prev_call.subtrace
        get_gen_fn(prev_subtrace) === gen_fn || gen_fn_changed_error(key)
        (subtrace, weight, discard, retdiff) = fix_update(args, argdiff,
            prev_subtrace, constraints)
    else
        if !isempty(constraints)
            error("fix_update attempted to constrain addresses under new key: $key")
        end
        (subtrace, weight) = initialize(gen_fn, args, EmptyAssignment())
    end

    # update weight
    state.weight += weight

    # update discard
    if has_previous
        set_subassmt!(state.discard, key, discard)
    end

    # update calldiffs
    if has_previous
        if isnodiff(retdiff)
            set_leaf_node!(state.calldiffs, key, NoCallDiff())
        else
            set_leaf_node!(state.calldiffs, key, CustomCallDiff(retdiff))
        end
    else
        set_leaf_node!(state.calldiffs, key, NewCallDiff())
    end

    # add to the trace
    add_call!(state.trace, key, subtrace)

    # get return value
    retval = get_retval(subtrace)

    retval
end

function splice(state::GFFixUpdateState, gen_fn::DynamicDSLFunction,
                args::Tuple)
    prev_params = state.params
    state.params = gen_fn.params
    retval = exec(gen_fn, state, args)
    state.params = prev_params
    retval
end

function fix_delete_recurse(prev_calls::Trie{Any,CallRecord},
                            visited::EmptyAddressSet)
    noise = 0.
    for (key, call) in get_leaf_nodes(prev_calls)
        noise += call.noise
    end
    for (key, subcalls) in get_internal_nodes(prev_calls)
        noise += fix_delete_recurse(subcalls, EmptyAddressSet())
    end
    noise
end

function fix_delete_recurse(prev_calls::Trie{Any,CallRecord},
                            visited::DynamicAddressSet)
    noise = 0.
    for (key, call) in get_leaf_nodes(prev_calls)
        if !has_leaf_node(visited, key)
            noise += call.noise
        end
    end
    for (key, subcalls) in get_internal_nodes(prev_calls)
        if has_internal_node(visited, key)
            subvisited = get_internal_node(visited, key)
        else
            subvisited = EmptyAddressSet()
        end
        noise += fix_delete_recurse(subcalls, subvisited)
    end
    noise
end

function fix_update(args::Tuple, argdiff, trace::DynamicDSLTrace,
                    constraints::Assignment)
    gen_fn = trace.gen_fn
    state = GFFixUpdateState(gen_fn, args, argdiff, trace,
        constraints, gen_fn.params)
    retval = exec_for_update(gen_fn, state, args)
    set_retval!(state.trace, retval)

    state.weight -= fix_delete_recurse(trace.calls, get_visited(state.visitor))

    visited = get_visited(state.visitor)
    if !all_visited(visited, constraints)
        error("Did not visit all constraints")
    end

    (state.trace, state.weight, state.discard, state.retdiff)
end
