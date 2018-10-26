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
    calldiffs::HomogenousTrue{Any,Any}
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
get_choice_diff(state::GFUpdateState, addr) = get_leaf_node(state.choicediffs, addr)
get_call_diff(state::GFUpdateState, addr) = get_leaf_node(state.calldiffs, addr)

function addr(state::GFUpdateState, dist::Distribution{T}, args, addr) where {T}
    local retval::T
    local prev_retval::T

    # check that address was not already visited, and mark it as visited
    visit!(state.visitor, addr)

    # check for previous choice at this address
    has_previous = has_primitive_call(state.prev_trace, addr)
    if has_previous
        prev_retval = get_primitive_call(state.prev_trace, addr)
    end

    # check for constraints at this address
    constrained = has_leaf_node(state.constraints, addr)
    if has_internal_node(state.constraints, addr)
        lightweight_got_internal_node_err(addr)
    end
    
    # obtain return value from previous trace or constraints
    if constrained
        retval = get_leaf_node(state.constraints, addr)
    elseif has_previous
        retval = prev_retval
    else
        error("Constraint not given for new address: $addr")
    end

    # record the previous value as discarded if it is replaced
    if constrained && has_previous
        set_leaf_node!(state.discard, addr, prev_retval)
    end

    # choicediff
    if constrained && has_previous
        choicediff = PrevChoiceDiff(prev_retval)
    elseif has_previous
        choicediff = NoChoiceDiff()
    else
        choicediff = NewChoiceDiff()
    end
    set_leaf_node!(state.choicediffs, addr, choicediff)

    # update trace and score
    score = logpdf(dist, retval, args...)
    call = CallRecord(score, retval, args)
    state.trace = assoc_primitive_call(state.trace, addr, call)
    state.score += score

    return retval 
end

function addr(state::GFUpdateState, gen::Generator{T,U}, args, addr, argdiff) where {T,U}
    local prev_trace::U
    local trace::U
    local retval::T

    # check address was not already visited, and mark it as visited
    visit!(state.visitor, addr)

    # check for constraints at this address
    if has_internal_node(state.constraints, addr)
        constraints = get_internal_node(state.constraints, addr)
    elseif has_leaf_node(state.constraints, addr)
        lightweight_got_leaf_node_err(addr)
    else
        constraints = EmptyAssignment()
    end

    if has_subtrace(state.prev_trace, addr)

        # address already populated
        prev_trace = get_subtrace(state.prev_trace, addr)
        (trace, _, discard, retdiff) = update(gen, args, argdiff,
            prev_trace, constraints)
        set_internal_node!(state.discard, addr, discard)
        set_leaf_node!(state.calldiffs, addr, CustomCallDiff(retdiff))
    else

        # address is new
        trace = assess(gen, args, constraints)
        set_leaf_node!(state.calldiffs, addr, NewCallDiff())
    end

    # update trace and score
    retval = get_call_record(trace).retval
    state.trace = assoc_subtrace(state.trace, addr, trace)
    state.score += call.score

    return retval 
end

splice(state::GFUpdateState, gen::GenFunction, args::Tuple) = exec(gf, state, args)

function update(gen::GenFunction, new_args, argdiff, trace::GFTrace, constraints)
    state = Gen.GFUpdateState(argdiff, trace, constraints, gen.params)
    retval = Gen.exec(gen, state, new_args)
    new_call = Gen.CallRecord{Any}(state.score, retval, new_args)
    state.trace.call = new_call
    # discard addresses that were deleted
    unvisited = Gen.get_unvisited(state.visitor, get_assignment(state.prev_trace))
    merge!(state.discard, unvisited) # TODO use merge()?
    if !isempty(Gen.get_unvisited(state.visitor, constraints))
        error("Update did not consume all constraints")
    end
    
    # compute the weight
    prev_score = get_call_record(trace).score
    weight = state.score - prev_score
    (state.trace, weight, state.discard, state.retdiff)
end
