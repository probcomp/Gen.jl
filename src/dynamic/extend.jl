mutable struct GFExtendState
    prev_trace::DynamicDSLTrace
    trace::DynamicDSLTrace
    constraints::Any
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
end

function GFExtendState(gen_fn, args, prev_trace,
                       constraints, params)
    visitor = AddressVisitor()
    GFExtendState(prev_trace, DynamicDSLTrace(gen_fn, args), constraints,
        0., visitor, params)
end

function traceat(state::GFExtendState, dist::Distribution{T},
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
    !constrained && check_no_submap(state.constraints, key)
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

    # add to the trace
    add_choice!(state.trace, key, retval, score)

    # update weight
    if constrained
        state.weight += score
    elseif has_previous
        state.weight += score - prev_score
    end

    retval 
end

function traceat(state::GFExtendState, gen_fn::GenerativeFunction{T,U},
                 args, key) where {T,U}
    local prev_trace::U
    local trace::U
    local retval::T

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for constraints at this key
    check_no_value(state.constraints, key)
    constraints = get_submap(state.constraints, key)

    # get subtrace
    has_previous = has_call(state.prev_trace, key)
    if has_previous
        prev_call = get_call(state.prev_trace, key)
        prev_subtrace = prev_call.subtrace
        get_gen_fn(prev_subtrace) === gen_fn || gen_fn_changed_error(key)
        (subtrace, weight, _) = extend(prev_subtrace,
            args, map((_) -> UnknownChange(), args), constraints)
    else
        (subtrace, weight) = generate(gen_fn, args, constraints)
    end

    # update weight
    state.weight += weight

    # update trace
    add_call!(state.trace, key, subtrace)

    # get return value
    retval = get_retval(subtrace)

    retval 
end

function splice(state::GFExtendState, gen_fn::DynamicDSLFunction,
                args::Tuple)
    prev_params = state.params
    state.params = gen_fn.params
    retval = exec(gen_fn, state, args)
    state.params = prev_params
    retval
end

function extend(trace::DynamicDSLTrace, args::Tuple, argdiffs::Tuple,
                constraints::ChoiceMap)
    gen_fn = trace.gen_fn
    state = GFExtendState(gen_fn, args, trace,
        constraints, gen_fn.params)
    retval = exec(gen_fn, state, args)
    set_retval!(state.trace, retval)
    visited = get_visited(state.visitor)
    if !all_visited(visited, constraints)
        error("Did not visit all constraints")
    end
    (state.trace, state.weight, UnknownChange())
end
