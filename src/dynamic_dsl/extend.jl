mutable struct GFExtendState
    prev_trace::DynamicDSLTrace
    trace::DynamicDSLTrace
    constraints::Any
    score::Float64
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
    argdiff::Any
    retdiff::Any
    choicediffs::HomogenousTrie{Any,Any}
    calldiffs::HomogenousTrie{Any,Any}
end

function GFExtendState(argdiff, prev_trace, constraints, params)
    visitor = AddressVisitor()
    GFExtendState(prev_trace, DynamicDSLTrace(), constraints, 0., 0.,
        visitor, params, argdiff, DefaultRetDiff(),
        HomogenousTrie{Any,Any}(), HomogenousTrie{Any,Any}())
end

function addr(state::GFExtendState, gen::GenerativeFunction, args, key)
    addr(state, gen, args, key, UnknownArgDiff())
end


function addr(state::GFExtendState, dist::Distribution{T}, args, key) where {T}

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for previous choice at this key 
    has_previous = has_primitive_call(state.prev_trace, key)
    local prev_retval::T
    if has_previous
        prev_call = get_primitive_call(state.prev_trace, key)
        prev_retval = prev_call.retval
        @assert prev_call.args == args
    end

    # check for constraints at this key
    constrained = has_value(state.constraints, key)
    lightweight_check_no_subassmt(state.constraints, key)
    if has_previous && constrained
        error("Extend attempted to change value of random choice at $key")
    end

    # get return value and logpdf
    local retval::T
    if has_previous
        retval = prev_retval
    elseif constrained
        retval = get_value(state.constraints, key)
    else
        retval = random(dist, args...)
    end
    score = logpdf(dist, retval, args...)

    # update choicediffs
    if has_previous
        set_choice_diff!(state, key, constrained, prev_retval) 
    else
        set_choice_diff_no_prev!(state, key)
    end

    # update trace
    call = GFCallRecord(score, retval, args)
    state.trace = assoc_primitive_call(state.trace, key, call)

    # update score
    state.score += score
    
    # update weight
    if constrained
        state.weight += score
    end

    return retval 
end

function addr(state::GFExtendState, gen::GenerativeFunction{T,U}, args, key, argdiff) where {T,U}

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for constraints at this key
    lightweight_check_no_value(state.constraints, key)
    constraints = get_subassmt(state.constraints, key)

    # get subtrace
    has_previous = has_subtrace(state.prev_trace, key)
    local prev_trace::U
    local trace::U
    if has_previous
        prev_trace = get_subtrace(state.prev_trace, key)
        (trace, weight, retdiff) = extend(gen, args, argdiff, prev_trace, constraints)
    else
        (trace, weight) = initialize(gen, args, constraints)
    end

    # get return value
    local retval::T
    retval = get_call_record(trace).retval

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

    # update trace
    state.trace = assoc_subtrace(state.trace, key, trace)

    # update score
    state.score += get_call_record(trace).score

    # update weight
    state.weight += weight

    return retval 
end

splice(state::GFExtendState, gen::DynamicDSLFunction, args::Tuple) = exec_for_update(gf, state, args)

function extend(gf::DynamicDSLFunction, args::Tuple, argdiff,
                trace::DynamicDSLTrace, constraints)
    state = GFExtendState(argdiff, trace, constraints, gf.params)
    retval = exec_for_update(gf, state, args)
    call = GFCallRecord(state.score, retval, args)
    state.trace.call = call
    (state.trace, state.weight, state.retdiff)
end
