mutable struct GFFixUpdateState
    prev_trace::GFTrace
    trace::GFTrace
    constraints::Any
    score::Float64
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
    discard::DynamicAssignment
    argdiff::Any
    retdiff::Any
    choicediffs::HomogenousTrie{Any,Any}
    calldiffs::HomogenousTrie{Any,Any}
end

function GFFixUpdateState(argdiff, prev_trace, constraints, params)
    visitor = AddressVisitor()
    discard = DynamicAssignment()
    GFFixUpdateState(prev_trace, GFTrace(), constraints, 0., 0., visitor,
                     params, discard, argdiff, GenFunctionDefaultRetDiff(),
                     HomogenousTrie{Any,Any}(), HomogenousTrie{Any,Any}())
end

# TODO code dup
get_arg_diff(state::GFFixUpdateState) = state.argdiff
set_ret_diff!(state::GFFixUpdateState, value) = state.retdiff = value
get_choice_diff(state::GFFixUpdateState, key) = get_leaf_node(state.choicediffs, key)
get_call_diff(state::GFFixUpdateState, key) = get_leaf_node(state.calldiffs, key)

function addr(state::GFFixUpdateState, dist::Distribution{T}, args, key) where {T}

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for previous choice at this key 
    has_previous = has_primitive_call(state.prev_trace, key)
    local prev_retval::T
    if has_previous
        prev_call = get_primitive_call(state.prev_trace, key)
        prev_retval = prev_call.retval
        prev_score = prev_call.score
    end

    # check for constraints at this key
    constrained = has_leaf_node(state.constraints, key)
    lightweight_check_no_internal_node(state.constraints, key)
    
    if constrained && !has_previous
        error("fix_update attempted to constrain a new key: $key")
    end

    # record the previous value as discarded if it is replaced
    if constrained && has_previous
        set_leaf_node!(state.discard, key, prev_retval)
    end

    # obtain return value from previous trace, constraints, or by sampling
    local retval::T
    if constrained
        retval = get_leaf_node(state.constraints, key)
    elseif has_previous
        retval = prev_retval
    else
        retval = random(dist, args...)
    end

    # choicediff (TODO this is the same for force update; can reduce code duplication)
    if constrained && has_previous
        choicediff = PrevChoiceDiff(prev_retval)
    elseif has_previous
        choicediff = NoChoiceDiff()
    else
        choicediff = NewChoiceDiff()
    end
    set_leaf_node!(state.choicediffs, key, choicediff)

    # update trace and score
    score = logpdf(dist, retval, args...)
    call = CallRecord(score, retval, args)
    state.trace = assoc_primitive_call(state.trace, key, call)
    state.score += score

    # update weight
    if has_previous
        state.weight += score - prev_score
    end

    return retval 
end

function addr(state::GFFixUpdateState, gen::Generator, args, key)
    addr(state, gen, args, key, UnknownArgDiff())
end

function addr(state::GFFixUpdateState, gen::Generator{T,U}, args, key, argdiff) where {T,U}

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for constraints at this key 
    lightweight_check_no_leaf_node(state.constraints, key)
    if has_internal_node(state.constraints, key)
        constraints = get_internal_node(state.constraints, key)
    else
        constraints = EmptyAssignment()
    end

    # get new trace
    local prev_trace::U
    local trace::U
    if has_subtrace(state.prev_trace, key)

        # key already populated
        prev_trace = get_subtrace(state.prev_trace, key)
        (trace, weight, discard, retdiff) = fix_update(gen, args, argdiff,
            prev_trace, constraints)
        state.weight += weight
        set_internal_node!(state.discard, key, discard)
        set_leaf_node!(state.calldiffs, key, CustomCallDiff(retdiff))
    else

        # key is new
        if has_internal_node(state.constraints, key)
            error("fix_update attempted to constrain new key: $key")
        end
        trace = simulate(gen, args)
        set_leaf_node!(state.calldiffs, key, NewCallDiff())
    end

    # update trace and score
    local retval::T
    retval = get_call_record(trace).retval
    state.trace = assoc_subtrace(state.trace, key, trace)
    state.score += call.score
    
    return retval
end

splice(state::GFFixUpdateState, gen::GenFunction, args::Tuple) = exec_for_update(gf, state, args)

function fix_update(gf::GenFunction, args, argdiff, prev_trace::GFTrace, constraints)
    state = GFFixUpdateState(argdiff, prev_trace, constraints, gf.params)
    retval = exec_for_update(gf, state, args)
    new_call = CallRecord(state.score, retval, args)
    state.trace.call = new_call
    unconsumed = get_unvisited(state.visitor, constraints)
    if !isempty(unconsumed)
        error("Update did not consume all constraints")
    end
    (state.trace, state.weight, state.discard, state.retchange)
end
