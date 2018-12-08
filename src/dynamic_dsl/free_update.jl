mutable struct GFFreeUpdateState
    prev_trace::DynamicDSLTrace
    trace::DynamicDSLTrace
    selection::AddressSet
    score::Float64
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
    argdiff::Any
    retdiff::Any
    choicediffs::HomogenousTrie{Any,Any}
    calldiffs::HomogenousTrie{Any,Any}
end

function GFFreeUpdateState(argdiff, prev_trace, selection, params)
    visitor = AddressVisitor()
    GFFreeUpdateState(prev_trace, DynamicDSLTrace(), selection, 0., 0., visitor,
                      params, argdiff, DefaultRetDiff(),
                      HomogenousTrie{Any,Any}(), HomogenousTrie{Any,Any}())
end

function addr(state::GFFreeUpdateState, dist::Distribution{T}, args, key) where {T}

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

    # check whether the key was selected
    if has_internal_node(state.selection, key)
        error("Got internal node but expected leaf node in selection at $key")
    end
    in_selection = has_leaf_node(state.selection, key)

    # get return value and logpdf
    local retval::T
    if has_previous && in_selection
        # there was a previous value, and it was in the selection
        # simulate a new value; it does not contribute to the weight
        retval = random(dist, args...)
    elseif has_previous
        # there was a previous value, and it was not in the selection
        # use the previous value; it contributes to the weight
        retval = prev_retval
    else
        # there is no previous value
        # simulate a new valuel; it does not contribute to the weight
        retval = random(dist, args...)
    end
    score = logpdf(dist, retval, args...)

    # update choicediffs
    if has_previous
        set_choice_diff!(state, key, in_selection, prev_retval) 
    else
        set_choice_diff_no_prev!(state, key)
    end

    # update trace
    call = GFCallRecord(score, retval, args)
    state.trace = assoc_primitive_call(state.trace, key, call)

    # update score
    state.score += score

    # update weight
    if has_previous && !in_selection
        state.weight += score - prev_score
    end

    return retval
end

function addr(state::GFFreeUpdateState, gen::GenerativeFunction, args, key)
    addr(state, gen, args, key, UnknownArgDiff())
end

function addr(state::GFFreeUpdateState, gen::GenerativeFunction{T,U}, args, key, argdiff) where {T,U}

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check whether the key was selected
    if has_leaf_node(state.selection, key)
        error("Entire sub-traces cannot be selected, tried to select $key")
    end
    if has_internal_node(state.selection, key)
        selection = get_internal_node(state.selection, key)
    else
        selection = EmptyAddressSet()
    end

    # get subtrace
    local prev_retval::T
    local trace::U
    has_previous = has_subtrace(state.prev_trace, key)
    if has_previous
        prev_trace = get_subtrace(state.prev_trace, key) 
        (trace, weight, retdiff) = free_update(
            gen, args, argdiff, prev_trace, selection)
    else
        trace = simulate(gen, args)
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
    if has_previous
        state.weight += weight
    end

    return retval
end

splice(state::GFFreeUpdateState, gf::DynamicDSLFunction, args::Tuple) = exec_for_update(gf, state, args)

function free_update(gen::DynamicDSLFunction, new_args::Tuple, argdiff,
                     trace::DynamicDSLTrace, selection::AddressSet)
    state = GFFreeUpdateState(argdiff, trace, selection, gen.params)
    retval = exec_for_update(gen, state, new_args)
    new_call = GFCallRecord(state.score, retval, new_args)
    state.trace.call = new_call
    (state.trace, state.weight, state.retdiff)
end
