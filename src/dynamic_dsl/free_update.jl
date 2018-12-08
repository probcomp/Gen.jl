mutable struct GFFreeUpdateState
    prev_trace::DynamicDSLTrace
    trace::DynamicDSLTrace
    selection::AddressSet
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
    argdiff::Any
    retdiff::Any
    choicediffs::HomogenousTrie{Any,Any}
    calldiffs::HomogenousTrie{Any,Any}
end

function GFFreeUpdateState(gen_fn, args, argdiff, prev_trace,
                           selection, params)
    visitor = AddressVisitor()
    GFFreeUpdateState(prev_trace, DynamicDSLTrace(gen_fn, args), selection,
        0., visitor, params, argdiff, DefaultRetDiff(),
        HomogenousTrie{Any,Any}(), HomogenousTrie{Any,Any}())
end

function addr(state::GFFreeUpdateState, dist::Distribution{T},
              args, key) where {T}
    local prev_retval::T
    local retval::T

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for previous choice at this key 
    has_previous = has_choice(state.prev_trace, key)
    if has_previous
        prev_call = get_choice(state.prev_trace, key)
        prev_retval = prev_call.retval
        prev_score = prev_call.score
    end

    # check whether the key was selected
    if has_internal_node(state.selection, key)
        error("Got internal node but expected leaf node in selection at $key")
    end
    in_selection = has_leaf_node(state.selection, key)

    # get return value
    if has_previous && in_selection
        retval = random(dist, args...)
    elseif has_previous
        retval = prev_retval
    else
        retval = random(dist, args...)
    end

    # compute logpdf
    score = logpdf(dist, retval, args...)

    # update choicediffs
    if has_previous
        set_choice_diff!(state, key, in_selection, prev_retval) 
    else
        set_choice_diff_no_prev!(state, key)
    end

    # update weight
    if has_previous && !in_selection
        state.weight += score - prev_score
    end

    # add to the trace
    add_choice!(state.trace, ChoiceRecord(retval, score))

    retval
end

function addr(state::GFFreeUpdateState, gen_fn::GenerativeFunction, args, key)
    addr(state, gen_fn, args, key, UnknownArgDiff())
end

function addr(state::GFFreeUpdateState, gen_fn::GenerativeFunction{T,U},
              args, key, argdiff) where {T,U}
    local prev_retval::T
    local trace::U
    local retval::T

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
    has_previous = has_call(state.prev_trace, key)
    if has_previous
        prev_call = get_call(state.prev_trace, key)
        prev_subtrace = prev_call.subtrace
        (subtrace, weight, retdiff) = free_update(
            gen_fn, args, argdiff, prev_subtrace, selection)
    else
        (subtrace, weight) = initialize(gen_fn, args, EmptyAssignment())
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

    # add to the trace
    add_call!(state.trace, key, subtrace)

    # get return value
    retval = get_retval(subtrace)

    retval
end

function splice(state::GFFreeUpdateState, gen_fn::DynamicDSLFunction,
                args::Tuple)
    exec_for_update(gen_fn, state, args)
end

function free_delete_recurse(prev_calls::HomogeneousTrie{Any,CallRecord},
                            visited::EmptyAddressSet)
    noise = 0.
    for (key, call) in get_leaf_nodes(prev_calls)
        noise += call.noise
    end
    for (key, subcalls) in get_internal_nodes(prev_calls)
        noise += free_delete_recurse(subcalls, EmptyAddressSet())
    end
    noise
end

function free_delete_recurse(prev_calls::HomogeneousTrie{Any,CallRecord},
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
        noise += free_delete_recurse(subcalls, subvisited)
    end
    noise
end

function free_update(gen_fn::DynamicDSLFunction, args::Tuple, argdiff,
                     trace::DynamicDSLTrace, selection::AddressSet)
    @assert gen_fn === trace.gen_fn
    state = GFFreeUpdateState(gen_fn, args, argdiff, trace,
        selection, gen.params)
    retval = exec_for_update(gen, state, args)
    set_retval!(state.trace, retval)

    visited = state.visitor.visited
    state.weight -= free_delete_recurse(trace.calls, visited)

    (state.trace, state.weight, state.retdiff)
end
