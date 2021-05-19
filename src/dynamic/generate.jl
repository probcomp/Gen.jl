mutable struct GFGenerateState
    trace::DynamicDSLTrace
    constraints::ChoiceMap
    weight::Float64
    visitor::AddressVisitor
    active_gen_fn::DynamicDSLFunction # mutated by splicing
    parameter_context::Dict

    function GFGenerateState(gen_fn, args, constraints, parameter_context)
        parameter_store = get_julia_store(parameter_context)
        registered_julia_parameters = get_parameters(gen_fn, parameter_context)[parameter_store]
        trace = DynamicDSLTrace(
            gen_fn, args, parameter_store, parameter_context, registered_julia_parameters)
        return new(trace, constraints, 0., AddressVisitor(), gen_fn, parameter_context)
    end
end

get_parameter_store(state::GFGenerateState) = get_parameter_store(state.trace)

get_parameter_id(state::GFGenerateState, name::Symbol) = (state.active_gen_fn, name)

get_active_gen_fn(state::GFGenerateState) = state.active_gen_fn

function set_active_gen_fn!(state::GFGenerateState, gen_fn::GenerativeFunction)
    state.active_gen_fn = gen_fn
end

function traceat(state::GFGenerateState, dist::Distribution{T},
              args, key) where {T}
    local retval::T

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for constraints at this key
    constrained = has_value(state.constraints, key)
    !constrained && check_no_submap(state.constraints, key)

    # get return value
    if constrained
        retval = get_value(state.constraints, key)
    else
        retval = random(dist, args...)
    end

    # compute logpdf
    score = logpdf(dist, retval, args...)

    # add to the trace
    add_choice!(state.trace, key, retval, score)

    # increment weight
    if constrained
        state.weight += score
    end

    return retval
end

function traceat(state::GFGenerateState, gen_fn::GenerativeFunction{T,U},
              args, key) where {T,U}
    local subtrace::U
    local retval::T

    # check key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for constraints at this key
    constraints = get_submap(state.constraints, key)

    # get subtrace
    (subtrace, weight) = generate(
        gen_fn, args, constraints, state.parameter_context)

    # add to the trace
    add_call!(state.trace, key, subtrace)

    # update weight
    state.weight += weight

    # get return value
    retval = get_retval(subtrace)

    return retval
end

function generate(
        gen_fn::DynamicDSLFunction, args::Tuple, constraints::ChoiceMap,
        parameter_context::Dict)
    state = GFGenerateState(gen_fn, args, constraints, parameter_context)
    retval = exec(gen_fn, state, args)
    set_retval!(state.trace, retval)
    return (state.trace, state.weight)
end
