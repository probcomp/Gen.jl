mutable struct GFRegenerateState
    prev_trace::GFTrace
    trace::GFTrace
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

function GFRegenerateState(argdiff, prev_trace, selection, params)
    visitor = AddressVisitor()
    GFRegenerateState(prev_trace, GFTrace(), selection, 0., 0., visitor,
                      params, discard, argdiff, GenFunctionDefaultRetDiff(),
                      HomogenousTrie{Any,Any}(), HomogenousTrie{Any,Any}())
end

# TODO code dup
get_arg_diff(state::GFRegenerateState) = state.argdiff
set_ret_diff!(state::GFRegenerateState, value) = state.retdiff = value
get_choice_diff(state::GFRegenerateState, key) = get_leaf_node(state.choicediffs, key)
get_call_diff(state::GFRegenerateState, key) = get_leaf_node(state.calldiffs, key)

function addr(state::GFRegenerateState, dist::Distribution{T}, args, key) where {T}

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

    local retval::T
    if has_previous && !in_selection
        # there was a previous value, and it was not in the selection
        # use the previous value
        # it contributes to the weight
        prev_call = get_primitive_call(state.prev_trace, key)
        retval = prev_call.retval
        score = logpdf(dist, retval, args...)
        state.weight += score - prev_call.score
        retdiff = NoChoiceDiff()
    elseif has_previous
        # there was a previous value, and it was in the selection
        # simulate a new value
        # it does not contribute to the weight
        retval = random(dist, args...)
        score = logpdf(dist, retval, args...)
        prev_call = get_primitive_call(state.prev_trace, key)
        retdiff = PrevChoicediff(prev_call.retval)
    else
        # there is no previous value
        # simulate a new value
        # it does not contribute to the weight
        retval = random(dist, args...)
        score = logpdf(dist, retval, args...)
        retdiff = NewChoiceDiff()
    end
    set_leaf_node!(state.choicediffs, key, choicediff)
    call = CallRecord(score, retval, args)
    state.trace = assoc_primitive_call(state.trace, key, call)
    state.score += score
    retval
end

function addr(state::GFRegenerateState, gen::Generator, args, key)
    addr(state, gen, args, key, UnknownArgDiff())
end

function addr(state::GFRegenerateState, gen::Generator{T}, args, addr, argdiff) where {T}

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for constraints at this key 
    lightweight_check_no_leaf_node(state.constraints, key)
    if has_internal_node(state.constraints, key)
        constraints = get_internal_node(state.constraints, key)
    else
        constraints = EmptyAssignment()
    end

    has_previous = has_subtrace(state.prev_trace, addr)
    if has_previous
        if addr in state.selection
            error("Entire sub-traces cannot be selected, tried to select $addr")
        end
        prev_trace = get_subtrace(state.prev_trace, addr) 
        prev_call = get_call_record(prev_trace)
        if has_internal_node(state.selection, addr)
            selection = get_internal_node(state.selection, addr)
        else
            selection = EmptyAddressSet()
        end
        (trace, weight, retdiff) = regenerate(
            gen, args, argdiff, prev_trace, selection)
        state.weight += weight
        set_leaf_node!(state.calldiffs, key, CustomCallDiff(retdiff))
    else
        trace = simulate(gen, args)
        set_leaf_node!(state.calldiffs, key, NewCallDiff())
    end
    call = get_call_record(trace)
    state.score += call.score
    state.trace = assoc_subtrace(state.trace, addr, trace)
    call.retval::T
end

splice(state::GFRegenerateState, gf::GenFunction, args::Tuple) = exec_for_update(gf, state, args)

function regenerate(gen::GenFunction, new_args, argdiff, trace, selection)
    state = GFRegenerateState(argdiff, trace, selection, gen.params)
    retval = exec_for_update(gen, state, new_args)
    new_call = CallRecord(state.score, retval, new_args)
    state.trace.call = new_call
    (state.trace, state.weight, state.retchange)
end
