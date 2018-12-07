mutable struct GFInitializeState
    trace::DynamicDSLTrace
    constraints::Assignment
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
end

function GFInitializeState(gen_fn, args, constraints, params)
    trace = DynamicDSLTrace(gen_fn, args)
    GFInitializeState(trace, constraints, 0., AddressVisitor(), params)
end

function addr(state::GFInitializeState, dist::Distribution{T},
              args, key) where {T}
    local retval::T

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for constraints at this key
    constrained = has_value(state.constraints, key)
    !constrained && check_no_subassmt(state.constraints, key)

    # get return value
    if constrained
        retval = get_value(state.constraints, key)
    else
        retval = random(dist, args...)
    end

    # compute logpdf
    score = logpdf(dist, retval, args...)

    # add to the trace
    add_choice!(state.trace, key, ChoiceRecord(retval, score))

    # increment weight
    if constrained
        state.weight += score
    end

    retval
end

function addr(state::GFInitializeState, gen_fn::GenerativeFunction{T,U},
              args, key) where {T,U}
    local subtrace::U
    local retval::T

    # check key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for constraints at this key
    constraints = get_subassmt(state.constraints, key)

    # get subtrace
    (subtrace, weight) = initialize(gen_fn, args, constraints)

    # add to the trace
    add_call!(state.trace, key, subtrace)

    # update weight
    state.weight += weight

    # get return value
    retval = get_retval(subtrace)

    retval
end

function splice(state::GFInitializeState, gen_fn::DynamicDSLFunction,
                args::Tuple)
    exec(gen_fn, state, args)
end

function initialize(gen_fn::DynamicDSLFunction, args::Tuple,
                    constraints::Assignment)
    state = GFInitializeState(gen_fn, args, constraints, gen_fn.params)
    retval = exec(gen_fn, state, args) 
    set_retval!(state.trace, retval)
    (state.trace, state.weight)
end
