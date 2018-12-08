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
    choicediffs::HomogenousTrie{Any,Any}
    calldiffs::HomogenousTrie{Any,Any}
end

function GFFixUpdateState(gen_fn, args, argdiff, prev_trace,
                          constraints, params)
    visitor = AddressVisitor()
    discard = DynamicAssignment()
    GFFixUpdateState(prev_trace, DynamicDSLTrace(gen_fn, args), constraints,
        0., visitor, params, discard, argdiff, DefaultRetDiff(),
        HomogenousTrie{Any,Any}(), HomogenousTrie{Any,Any}())
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
        prev_retval = prev_call.retval
        prev_score = prev_call.score
    end

    # check for constraints at this key
    constrained = has_value(state.constraints, key)
    lightweight_check_no_subassmt(state.constraints, key)
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
    add_choice!(state.trace, ChoiceRecord(retval, score))

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
    lightweight_check_no_value(state.constraints, key)
    constraints = get_subassmt(state.constraints, key)

    # get subtrace
    has_previous = has_subtrace(state.prev_trace, key)
    if has_previous
        prev_trace = get_subtrace(state.prev_trace, key)
        (subtrace, weight, discard, retdiff) = fix_update(gen_fn, args, argdiff,
            prev_trace, constraints)
    else
        if !isempty(constraints)
            error("fix_update attempted to constrain addresses under new key: $key")
        end

        # p(t; x) / q(t; x) -- could be 1 or not?
        # p(r, t; x) / (q(t; x) q(r; t, x)) -- generally will not be one
        # this is called 'noise', note that the noise is integrated into our
        # cumulative noise value in add_call!
        (subtrace, weight) = initialize(gen_fn, args, EmptyAssignment())
    end

    # update weight
    if has_previous
        state.weight += weight
    end

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
    exec_for_update(gen_fn, state, args)
end

function fix_delete_recurse(prev_calls::HomogeneousTrie{Any,CallRecord},
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

function fix_delete_recurse(prev_calls::HomogeneousTrie{Any,CallRecord},
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

function fix_update(gen_fn::DynamicDSLFunction, args, argdiff,
                    trace::DynamicDSLTrace, constraints)
    state = GFFixUpdateState(argdiff, prev_trace, constraints, gf.params)
    retval = exec_for_update(gf, state, args)
    @assert gen_fn === trace.gen_fn
    state = GFFixUpdateState(gen_fn, args, argdiff, trace,
        constraints, gen.params)
    retval = exec_for_update(gen, state, args)
    set_retval!(state.trace, retval)

    visited = state.visitor.visited
    state.weight -= fix_delete_recurse(state.prev_trace.calls, visited)

    if !all_visited(visited, constraints)
        error("Did not visit all constraints")
    end

    (state.trace, state.weight, state.discard, state.retdiff)
end
