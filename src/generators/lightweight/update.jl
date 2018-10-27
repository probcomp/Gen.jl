mutable struct GFUpdateState
    prev_trace::GFTrace
    trace::GFTrace
    constraints::Any
    score::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
    discard::DynamicAssignment
    argdiff::Any
    retdiff::Any
    choicediffs::HomogenousTrie{Any,Any}
    calldiffs::HomogenousTrie{Any,Any}
end

function GFUpdateState(argdiff, prev_trace, constraints, params)
    visitor = AddressVisitor()
    discard = DynamicAssignment()
    GFUpdateState(prev_trace, GFTrace(), constraints, 0., visitor,
                  params, discard, argdiff, GenFunctionDefaultRetDiff(),
                  HomogenousTrie{Any,Any}(), HomogenousTrie{Any,Any}())
end

get_arg_diff(state::GFUpdateState) = state.argdiff
set_ret_diff!(state::GFUpdateState, value) = state.retdiff = value
get_choice_diff(state::GFUpdateState, key) = get_leaf_node(state.choicediffs, key)
get_call_diff(state::GFUpdateState, key) = get_leaf_node(state.calldiffs, key)

function addr(state::GFUpdateState, dist::Distribution{T}, args, key) where {T}

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for previous choice at this key
    has_previous = has_primitive_call(state.prev_trace, key)
    local prev_retval::T
    if has_previous
        prev_retval = get_primitive_call(state.prev_trace, key).retval
    end

    # check for constraints at this key
    constrained = has_leaf_node(state.constraints, key)
    lightweight_check_no_internal_node(state.constraints, key)
    
    # obtain return value from previous trace or constraints
    local retval::T
    if constrained
        retval = get_leaf_node(state.constraints, key)
    elseif has_previous
        retval = prev_retval
    else
        error("Constraint not given for new key: $key")
    end

    # record the previous value as discarded if it is replaced
    if constrained && has_previous
        set_leaf_node!(state.discard, key, prev_retval)
    end

    # choicediff
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

    return retval 
end

function addr(state::GFUpdateState, gen::Generator, args, key)
    addr(state, gen, args, key, UnknownArgDiff())
end

function addr(state::GFUpdateState, gen::Generator{T,U}, args, key, argdiff) where {T,U}

    # check key was not already visited, and mark it as visited
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
        (trace, _, discard, retdiff) = update(gen, args, argdiff,
            prev_trace, constraints)
        set_internal_node!(state.discard, key, discard)
        set_leaf_node!(state.calldiffs, key, CustomCallDiff(retdiff))
    else

        # key is new
        trace = assess(gen, args, constraints)
        set_leaf_node!(state.calldiffs, key, NewCallDiff())
    end

    # update trace and score
    local retval::T
    retval = get_call_record(trace).retval
    state.trace = assoc_subtrace(state.trace, key, trace)
    state.score += call.score

    return retval 
end

splice(state::GFUpdateState, gen::GenFunction, args::Tuple) = exec_for_update(gf, state, args)

function update(gen::GenFunction, new_args, argdiff, trace::GFTrace, constraints)
    state = GFUpdateState(argdiff, trace, constraints, gen.params)
    retval = exec_for_update(gen, state, new_args)
    new_call = CallRecord{Any}(state.score, retval, new_args)
    state.trace.call = new_call
    # discard addresses that were deleted
    unvisited = get_unvisited(state.visitor, get_assignment(state.prev_trace))
    merge!(state.discard, unvisited) # TODO use merge()?
    if !isempty(get_unvisited(state.visitor, constraints))
        error("Update did not consume all constraints")
    end
    
    # compute the weight
    prev_score = get_call_record(trace).score
    weight = state.score - prev_score
    (state.trace, weight, state.discard, state.retdiff)
end
