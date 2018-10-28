##########################################
# argdiff, choicediff, calldiff, retdiff #
##########################################

struct GenFunctionDefaultRetDiff end
isnoretdiff(::GenFunctionDefaultRetDiff) = false

export GenFunctionDefaultRetDiff

get_arg_diff(state) = state.argdiff
set_ret_diff!(state, value) = state.retdiff = value
get_choice_diff(state, key) = get_leaf_node(state.choicediffs, key)
get_call_diff(state, key) = get_leaf_node(state.calldiffs, key)

function set_choice_diff_no_prev!(state, key::Int)
    choicediff = NewChoiceDiff()
    set_leaf_node!(state.choicediffs, key, choicediff)
end

function set_choice_diff!(state, key::Int, value_changed::Bool,
                          prev_retval::T) where {T}
    if value_changed
        choicediff = PrevChoiceDiff(prev_retval)
    else
        choicediff = NoChoiceDiff()
    end
    set_leaf_node!(state.choicediffs, key, choicediff)
end



################
# force update #
################

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
    if has_previous
        set_choice_diff!(state, key, constrained, prev_retval) 
    else
        set_choice_diff_no_prev!(state, key)
    end

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
    state.score += get_call_record(trace).score

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
                     params, discard, argdiff, GenFunctionDefaultRetDiff(),
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

    # choicediff
    if has_previous
        set_choice_diff!(state, key, constrained, prev_retval) 
    else
        set_choice_diff_no_prev!(state, key)
    end

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
    state.score += get_call_record(trace).score
    
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
    (state.trace, state.weight, state.discard, state.retdiff)
end


###############
# free update #
###############

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
                      params, argdiff, GenFunctionDefaultRetDiff(),
                      HomogenousTrie{Any,Any}(), HomogenousTrie{Any,Any}())
end

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

    # obtain return value from previous trace, or by sampling
    local retval::T
    if has_previous && in_selection
        # there was a previous value, and it was in the selection
        # simulate a new value
        # it does not contribute to the weight
        retval = random(dist, args...)
    elseif has_previous
        # there was a previous value, and it was not in the selection
        # use the previous value
        # it contributes to the weight
        prev_call = get_primitive_call(state.prev_trace, key)
        retval = prev_call.retval
    else
        # there is no previous value
        # simulate a new value
        # it does not contribute to the weight
        retval = random(dist, args...)
    end

    # choicediff
    if has_previous
        set_choice_diff!(state, key, in_selection, prev_retval) 
    else
        set_choice_diff_no_prev!(state, key)
    end

    # update trace, score, and weight
    score = logpdf(dist, retval, args...)
    if has_previous && !in_selection
        state.weight += score - prev_call.score
    end
    state.score += score
    call = CallRecord(score, retval, args)
    state.trace = assoc_primitive_call(state.trace, key, call)

    return retval
end

function addr(state::GFRegenerateState, gen::Generator, args, key)
    addr(state, gen, args, key, UnknownArgDiff())
end

function addr(state::GFRegenerateState, gen::Generator{T}, args, key, argdiff) where {T}

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for previous 
    has_previous = has_subtrace(state.prev_trace, key)
    if has_previous
        if key in state.selection
            error("Entire sub-traces cannot be selected, tried to select $key")
        end
        prev_trace = get_subtrace(state.prev_trace, key) 
        prev_call = get_call_record(prev_trace)
        if has_internal_node(state.selection, key)
            selection = get_internal_node(state.selection, key)
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
    state.trace = assoc_subtrace(state.trace, key, trace)
    call.retval::T
end

splice(state::GFRegenerateState, gf::GenFunction, args::Tuple) = exec_for_update(gf, state, args)

function regenerate(gen::GenFunction, new_args, argdiff, trace, selection)
    state = GFRegenerateState(argdiff, trace, selection, gen.params)
    retval = exec_for_update(gen, state, new_args)
    new_call = CallRecord(state.score, retval, new_args)
    state.trace.call = new_call
    (state.trace, state.weight, state.retdiff)
end


##########
# extend #
##########

mutable struct GFExtendState
    prev_trace::GFTrace
    trace::GFTrace
    args_change::Any
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
    GFExtendState(prev_trace, GFTrace(), args_change, constraints, 0., 0.,
        visitor, params, argdiff, GenFunctionDefaultRetDiff(),
        HomogenousTrie{Any,Any}(), HomogenousTrie{Any,Any}())
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
        prev_score = prev_call.score
    end

    # check for constraints at this key
    constrained = has_leaf_node(state.constraints, key)
    lightweight_check_no_internal_node(state.constraints, key)
    if has_previous && constrained
        error("Attempted to change value of random choice at $key during extend")
    end

    # obtain return value from previous trace, constraints, or by sampling
    local retval::T
    if has_previous
        retval = prev_retval
        score = logpdf(dist, retval, args...)
        state.weight += score - prev_score
    elseif constrained
        retval = get_leaf_node(state.constraints, key)
        score = logpdf(dist, retval, args...)
        state.weight += score
    else
        retval = random(dist, args...)
        score = logpdf(dist, retval, args...)
    end
    call = CallRecord(score, retval, args)

    # update trace, score, and weight
    state.trace = assoc_primitive_call(state.trace, key, call)
    state.score += score
    if has_previous
        state.weight += score - prev_score
    elseif constrained
        state.weight += score
    end

    # choicediff
    if has_previous
        set_choice_diff!(state, key, in_selection, prev_retval) 
    else
        set_choice_diff_no_prev!(state, key)
    end

    return retval 
end

function addr(state::GFExtendState, gen::Generator{T}, args, key, argdiff) where {T}

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for constraints at this key
    lightweight_check_no_leaf_node(state.constraints, addr)
    if has_internal_node(state.constraints, addr)
        constraints = get_internal_node(state.constraints, addr)
    else
        constraints = EmptyAssignment()
    end

    # get new trace
    local prev_trace::U
    local trace::U
    if has_subtrace(state.prev_trace, addr)

        # key already populated
        prev_trace = get_subtrace(state.prev_trace, addr)
        (trace, weight, retdiff) = extend(gen, args, argdiff, prev_trace, constraints)
        set_leaf_node!(state.calldiffs, key, CustomCallDiff(retdiff))
    else

        # key is new
        (trace, weight) = generate(gen, args, constraints)
    end

    # update trace, score, and weight
    local retval::T
    retval = get_call_record(trace).retval
    state.trace = assoc_subtrace(state.trace, addr, trace)
    state.score += call.score
    state.weight += weight

    return retval 
end

splice(state::GFExtendState, gen::GenFunction, args::Tuple) = exec_for_update(gf, state, args)

function extend(gf::GenFunction, args, argdiff, trace::GFTrace, constraints)
    state = GFExtendState(argdiff, trace, constraints, gf.params)
    retval = exec_for_update(gf, state, args)
    call = CallRecord(state.score, retval, args)
    state.trace.call = call
    (state.trace, state.weight, state.retdiff)
end
