mutable struct GFForceUpdateState
    prev_trace::DynamicDSLTrace
    trace::DynamicDSLTrace
    constraints::Any
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
    discard::DynamicAssignment
    argdiff::Any
    retdiff::Any
    choicediffs::HomogenousTrie{Any,Any}
    calldiffs::HomogenousTrie{Any,Any}
end

function GFForceUpdateState(gen_fn, args, argdiff, prev_trace,
                            constraints, params)
    visitor = AddressVisitor()
    discard = DynamicAssignment()
    GFForceUpdateState(prev_trace, DynamicDSLTrace(gen_fn, args), constraints,
        0., visitor, params, discard, argdiff, DefaultRetDiff(),
        HomogenousTrie{Any,Any}(), HomogenousTrie{Any,Any}())
end

function addr(state::GFForceUpdateState, dist::Distribution{T}, 
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
        prev_score = prev_call.score 
    end

    # check for constraints at this key
    constrained = has_value(state.constraints, key)
    lightweight_check_no_subassmt(state.constraints, key)

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
        error("Constraint not given for new key: $key")
    end

    # compute logpdf
    score = logpdf(dist, retval, args...)

    # update the weight
    if has_previous
        state.weight += score - prev_score
    end

    # update choicediffs
    if has_previous
        set_choice_diff!(state, key, constrained, prev_retval) 
    else
        set_choice_diff_no_prev!(state, key)
    end

    # add to the trace
    add_choice!(state.trace, ChoiceRecord(retval, score))

    retval 
end

function addr(state::GFForceUpdateState, gen_fn::GenerativeFunction, args, key)
    addr(state, gen_fn, args, key, UnknownArgDiff())
end

function addr(state::GFForceUpdateState, gen_fn::GenerativeFunction{T,U},
              args, key, argdiff) where {T,U}
    local prev_subtrace::U
    local subtrace::U
    local retval::T

    # check key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for constraints at this key
    lightweight_check_no_value(state.constraints, key)
    constraints = get_subassmt(state.constraints, key)

    # get subtrace
    has_previous = has_call(state.prev_trace, key)
    if has_previous
        prev_call = get_call(state.prev_trace, key)
        prev_subtrace = prev_call.subtrace
        (subtrace, weight, discard, retdiff) = force_update(gen_fn, args, argdiff,
            prev_subtrace, constraints)
    else
        (subtrace, weight) = initialize(gen_fn, args, constraints)
    end
    
    # update the weight
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

function splice(state::GFForceUpdateState, gen_fn::DynamicDSLFunction,
                args::Tuple)
    exec_for_update(gen_fn, state, args)
end

function force_delete_recurse(prev_choices::HomogeneousTrie{Any,ChoiceRecord},
                                     visited::EmptyAddressSet)
    score = 0.
    for (key, choice) in get_leaf_nodes(prev_choices)
        score += choice.score
    end
    for (key, subchoices) in get_internal_nodes(prev_choices)
        score += force_delete_recurse(subchoices, EmptyAddressSet())
    end
    score
end

function force_delete_recurse(prev_choices::HomogeneousTrie{Any,ChoiceRecord},
                                     visited::DynamicAddressSet)
    score = 0.
    for (key, choice) in get_leaf_nodes(prev_choices)
        if !has_leaf_node(visited, key)
            score += choice.score
        end
    end
    for (key, subchoices) in get_internal_nodes(prev_choices)
        if has_internal_node(visited, key)
            subvisited = get_internal_node(visited, key)
        else
            subvisited = EmptyAddressSet()
        end
        score += force_delete_recurse(subchoices, subvisited)
    end
    score
end

function force_delete_recurse(prev_calls::HomogeneousTrie{Any,CallRecord},
                        visited::EmptyAddressSet)
    score = 0.
    for (key, call) in get_leaf_nodes(prev_calls)
        score += call.score
    end
    for (key, subcalls) in get_internal_nodes(prev_calls)
        score += force_delete_recurse(subcalls, EmptyAddressSet())
    end
    score
end

function force_delete_recurse(prev_calls::HomogeneousTrie{Any,CallRecord},
                        visited::DynamicAddressSet)
    score = 0.
    for (key, call) in get_leaf_nodes(prev_calls)
        if !has_leaf_node(visited, key)
            score += call.score
        end
    end
    for (key, subcalls) in get_internal_nodes(prev_calls)
        if has_internal_node(visited, key)
            subvisited = get_internal_node(visited, key)
        else
            subvisited = EmptyAddressSet()
        end
        score += force_delete_recurse(subcalls, subvisited)
    end
    score
end

function add_unvisited_to_discard!(discard::DynamicAssignment,
                                   visited::DynamicAddressSet,
                                   prev_assmt::Assignment)
    for (key, value) in get_values_shallow(prev_assmt)
        if !has_leaf_node(visited, key)
            @assert !has_value(discard, key)
            @assert isempty(get_subassmt(discard, key))
            set_value!(discard, key, value)
        end
    end
    for (key, subassmt) in get_subassmts_shallow(prev_assmt)
        @assert !has_value(discard, key)
        if has_leaf_node(visited, key)
            # the recursive call to force_update already handled the discard
            # for this entire subassmt
            continue
        elseif has_internal_node(visited, key)
            subvisited = get_internal_node(visited, key)
            subdiscard = get_subassmt(discard, key)
            add_unvisited_to_discard!(subdiscard, subvisited, subassmt)
            set_subassmt!(discard, key, subdiscard)
        else
            # none of this subassmt was visited, so we discard the whole thing
            @assert isempty(get_subassmt(discard, key))
            set_subassmt!(discard, key, subassmt)
        end
    end
end

function force_update(gen_fn::DynamicDSLFunction, args, argdiff,
                      trace::DynamicDSLTrace, constraints)
    @assert gen_fn === trace.gen_fn
    state = GFForceUpdateState(gen_fn, args, argdiff, trace,
        constraints, gen.params)
    retval = exec_for_update(gen, state, args)
    set_retval!(state.trace, retval)

    visited = state.visitor.visited
    state.weight -= force_delete_recurse(trace.choices, visited)
    state.weight -= force_delete_recurse(trace.calls, visited)

    add_unvisited_to_discard!(state.discard, visited, get_assignment(trace))

    if !all_visited(visited, constraints)
        error("Did not visit all constraints")
    end
  
    (state.trace, state.weight, state.discard, state.retdiff)
end
