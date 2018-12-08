################
# force update #
################

mutable struct GFUpdateState
    prev_trace::GFTrace
    trace::GFTrace
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

function GFUpdateState(argdiff, prev_trace, constraints, params)
    visitor = AddressVisitor()
    discard = DynamicAssignment()
    GFUpdateState(prev_trace, GFTrace(), constraints, 0., visitor,
                  params, discard, argdiff, DefaultRetDiff(),
                  HomogenousTrie{Any,Any}(), HomogenousTrie{Any,Any}())
end

function addr(state::GFUpdateState, dist::Distribution{T}, args, key) where {T}

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for previous choice at this key
    has_previous = has_primitive_call(state.prev_trace, key)
    local prev_retval::T
    if has_previous
        prev_choice::ChoiceRecord = get_primitive_call(state.prev_trace, key)
        prev_retval = prev_choice.retval
        prev_score = prev_call.score 
    end

    # check for constraints at this key
    constrained = has_value(state.constraints, key)
    lightweight_check_no_subassmt(state.constraints, key)
    
    # get return value and logpdf
    local retval::T
    if constrained
        retval = get_value(state.constraints, key)
    elseif has_previous
        retval = prev_retval
    else
        error("Constraint not given for new key: $key")
    end
    score = logpdf(dist, retval, args...)

    # update the weight
    if has_previous
        state.weight += score - prev_score
    end

    # record the previous value as discarded if it is replaced
    if constrained && has_previous
        set_value!(state.discard, key, prev_retval)
    end

    # update choicediffs
    if has_previous
        set_choice_diff!(state, key, constrained, prev_retval) 
    else
        set_choice_diff_no_prev!(state, key)
    end

    # update trace
    state.trace.choices[key] = ChoiceRecord(retval, score)

    return retval 
end

function addr(state::GFUpdateState, gen::GenerativeFunction, args, key)
    addr(state, gen, args, key, UnknownArgDiff())
end

function addr(state::GFUpdateState, gen::GenerativeFunction{T,U}, args, key, argdiff) where {T,U}

    # check key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for constraints at this key
    lightweight_check_no_value(state.constraints, key)
    constraints = get_subassmt(state.constraints, key)

    # get subtrace
    local prev_trace::U
    local trace::U
    has_previous = has_subtrace(state.prev_trace, key)
    if has_previous
        prev_trace = get_subtrace(state.prev_trace, key)
        (trace, weight, discard, retdiff) = force_update(gen, args, argdiff,
            prev_trace, constraints)
    else
        (trace, weight) = initialize(gen, args, constraints)
    end
    
    # TODO what is the 'score'

    # update the weight
    state.weight += weight

    # get return value
    local retval::T
    retval = get_retval(trace)

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
 
    # update trace
    state.trace.calls[key] = CallRecord(trace, score)

    return retval 
end

splice(state::GFUpdateState, gen::DynamicDSLFunction, args::Tuple) = exec_for_update(gen, state, args)

function force_update(gen::DynamicDSLFunction, new_args, argdiff, trace::GFTrace, constraints)
    state = GFUpdateState(argdiff, trace, constraints, gen.params)
    retval = exec_for_update(gen, state, new_args)
    new_call = GFCallRecord{Any}(state.score, retval, new_args)
    state.trace.call = new_call

    # discard keys that were deleted
    unvisited = get_unvisited(state.visitor, get_assignment(state.prev_trace))
    merge!(state.discard, unvisited) # TODO use merge()?
    if !isempty(get_unvisited(state.visitor, constraints))
        error("Update did not consume all constraints")
    end
    
    (state.trace, state.weight, state.discard, state.retdiff)
end


##############
# fix update #
##############

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
                     params, discard, argdiff, DefaultRetDiff(),
                     HomogenousTrie{Any,Any}(), HomogenousTrie{Any,Any}())
end

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
    constrained = has_value(state.constraints, key)
    lightweight_check_no_subassmt(state.constraints, key)
    if constrained && !has_previous
        error("fix_update attempted to constrain a new key: $key")
    end

    # record the previous value as discarded if it is replaced
    if constrained && has_previous
        set_value!(state.discard, key, prev_retval)
    end

    # get return value and logpdf
    local retval::T
    if constrained
        retval = get_value(state.constraints, key)
    elseif has_previous
        retval = prev_retval
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
    if has_previous
        state.weight += score - prev_score
    end

    return retval 
end

function addr(state::GFFixUpdateState, gen::GenerativeFunction, args, key)
    addr(state, gen, args, key, UnknownArgDiff())
end

function addr(state::GFFixUpdateState, gen::GenerativeFunction{T,U}, args, key, argdiff) where {T,U}

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for constraints at this key 
    lightweight_check_no_value(state.constraints, key)
    constraints = get_subassmt(state.constraints, key)

    # get subtrace
    local prev_trace::U
    local trace::U
    has_previous = has_subtrace(state.prev_trace, key)
    if has_previous
        prev_trace = get_subtrace(state.prev_trace, key)
        (trace, weight, discard, retdiff) = fix_update(gen, args, argdiff,
            prev_trace, constraints)
    else
        if !isempty(get_subassmt(state.constraints, key))
            error("fix_update attempted to constrain addresses under new key: $key")
        end
        trace = simulate(gen, args)
    end

    # get return value
    local retval::T
    retval = get_call_record(trace).retval

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

    # update trace, score, and weight
    state.trace = assoc_subtrace(state.trace, key, trace)

    # update score
    state.score += get_call_record(trace).score

    # update weight
    if has_previous
        state.weight += weight
    end
    
    return retval
end

splice(state::GFFixUpdateState, gen::DynamicDSLFunction, args::Tuple) = exec_for_update(gf, state, args)

function fix_update(gf::DynamicDSLFunction, args, argdiff, prev_trace::GFTrace, constraints)
    state = GFFixUpdateState(argdiff, prev_trace, constraints, gf.params)
    retval = exec_for_update(gf, state, args)
    new_call = GFCallRecord(state.score, retval, args)
    state.trace.call = new_call
    unconsumed = get_unvisited(state.visitor, constraints)
    if !isempty(unconsumed)
        error("Update did not consume all constraints")
    end
    (state.trace, state.weight, state.discard, state.retdiff)
end


###############
# free update #
###############

mutable struct GFFreeUpdateState
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

function GFFreeUpdateState(argdiff, prev_trace, selection, params)
    visitor = AddressVisitor()
    GFFreeUpdateState(prev_trace, GFTrace(), selection, 0., 0., visitor,
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
                     trace::GFTrace, selection::AddressSet)
    state = GFFreeUpdateState(argdiff, trace, selection, gen.params)
    retval = exec_for_update(gen, state, new_args)
    new_call = GFCallRecord(state.score, retval, new_args)
    state.trace.call = new_call
    (state.trace, state.weight, state.retdiff)
end


##########
# extend #
##########

mutable struct GFExtendState
    prev_trace::GFTrace
    trace::GFTrace
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
    GFExtendState(prev_trace, GFTrace(), constraints, 0., 0.,
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
                trace::GFTrace, constraints)
    state = GFExtendState(argdiff, trace, constraints, gf.params)
    retval = exec_for_update(gf, state, args)
    call = GFCallRecord(state.score, retval, args)
    state.trace.call = call
    (state.trace, state.weight, state.retdiff)
end
