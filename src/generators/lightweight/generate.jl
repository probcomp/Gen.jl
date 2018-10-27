mutable struct GFGenerateState
    trace::GFTrace
    constraints::Any
    score::Float64
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
end

function GFGenerateState(constraints, params::Dict{Symbol,Any})
    GFGenerateState(GFTrace(), constraints, 0., 0., AddressVisitor(), params)
end

function addr(state::GFGenerateState, dist::Distribution{T}, args, addr) where {T}
    local retval::T

    # check that address was not already visited, and mark it as visited
    visit!(state.visitor, addr)

    # check for constraints at this address
    lightweight_check_no_internal_node(state.constraints, addr)
    constrained = has_leaf_node(state.constraints, addr)

    # either obtain return value from constraints, or sample one
    if constrained
        retval = get_leaf_node(state.constraints, addr)
    else
        retval = random(dist, args...)
    end

    # update trace, score, and weight
    score = logpdf(dist, retval, args...)
    call = CallRecord(score, retval, args)
    state.trace = assoc_primitive_call(state.trace, addr, call)
    state.trace.has_choices = true
    state.score += score
    if constrained
        state.weight += score
    end

    return retval
end

function addr(state::GFGenerateState, gen::Generator{T,U}, args, addr) where {T,U}
    local retval::T

    # check address was not already visited, and mark it as visited
    visit!(state.visitor, addr)

    # check for constraints at this address
    lightweight_check_no_leaf_node(state.constraints, addr)
    if has_internal_node(state.constraints, addr)
        constraints = get_internal_node(state.constraints, addr)
    else
        constraints = EmptyAssignment()
    end

    # generate subtrace and retrieve return value
    (trace::U, weight) = generate(gen, args, constraints)
    retval = get_call_record(trace).retval
    score = get_call_record(trace).score

    # update trace, score, and weight
    state.trace = assoc_subtrace(state.trace, addr, trace)
    state.trace.has_choices |= has_choices(trace)
    state.score += score
    state.weight += weight

    return retval
end

splice(state::GFGenerateState, gf::GenFunction, args::Tuple) = exec(gf, state, args)

function generate(gen::GenFunction, args, constraints)
    state = GFGenerateState(constraints, gen.params)
    retval = exec(gen, state, args) 
    # TODO add return type annotation for gen
    call = CallRecord{Any}(state.score, retval, args)
    state.trace.call = call
    (state.trace, state.weight)
end
