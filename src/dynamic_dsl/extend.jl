mutable struct GFExtendState
    prev_trace::DynamicDSLTrace
    trace::DynamicDSLTrace
    constraints::Any
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
    argdiff::Any
    retdiff::Any
    choicediffs::Trie{Any,Any}
    calldiffs::Trie{Any,Any}
end

function GFExtendState(gen_fn, args, argdiff, prev_trace,
                       constraints, params)
    visitor = AddressVisitor()
    GFExtendState(prev_trace, DynamicDSLTrace(gen_Fn, args), constraints,
        0., visitor, params, argdiff, DefaultRetDiff(),
        Trie{Any,Any}(), Trie{Any,Any}())
end

function addr(state::GFExtendState, dist::Distribution{T},
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
        prev_score = prev_choice.score
    end

    # check for constraints at this key
    constrained = has_value(state.constraints, key)
    !constrained && check_no_subassmt(state.constraints, key)
    if has_previous && constrained
        error("Extend attempted to change value of random choice at $key")
    end

    # get return value
    if has_previous
        retval = prev_retval
    elseif constrained
        retval = get_value(state.constraints, key)
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

    # add to the trace
    add_choice!(state.trace, key, ChoiceRecord(retval, score))

    # update weight
    if constrained
        state.weight += score
    end

    retval 
end

function addr(state::GFExtendState, gen_fn::GenerativeFunction, args, key)
    addr(state, gen_fn, args, key, UnknownArgDiff())
end

function addr(state::GFExtendState, gen_fn::GenerativeFunction{T,U},
              args, key, argdiff) where {T,U}
    local prev_trace::U
    local trace::U
    local retval::T

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for constraints at this key
    check_no_value(state.constraints, key)
    constraints = get_subassmt(state.constraints, key)

    # get subtrace
    has_previous = has_call(state.prev_trace, key)
    if has_previous
        prev_call = get_call(state.prev_trace, key)
        prev_subtrace = prev_call.subtrace
        (subtrace, weight, retdiff) = extend(gen_fn, args, argdiff,
            prev_subtrace, constraints)
    else
        (subtrace, weight) = initialize(gen_fn, args, constraints)
    end

    # update weight
    state.weight += weight

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
    add_call!(state.trace, key, subtrace)

    # get return value
    retval = get_retval(subtrace)

    retval 
end

function splice(state::GFExtendState, gen_fn::DynamicDSLFunction,
                args::Tuple)
    exec_for_update(gen_fn, state, args)
end

function extend(gen_fn::DynamicDSLFunction, args::Tuple, argdiff,
                trace::DynamicDSLTrace, constraints::Assignment)
    @assert gen_fn === trace.gen_fn
    state = GFExtendState(gen_fn, args, argdiff, trace,
        constraints, gf.params)
    retval = exec_for_update(gf, state, args)
    set_retval!(state.trace, retval)

    visited = get_visited(state.visitor)
    if !all_visited(visited, constraints)
        error("Did not visit all constraints")
    end

    (state.trace, state.weight, state.retdiff)
end
