mutable struct GFAssessState
    choices::ChoiceMap
    weight::Float64
    visitor::AddressVisitor
    active_gen_fn::DynamicDSLFunction # mutated by splicing
    parameter_context::Dict

    function GFAssessState(gen_fn, choices, parameter_context)
        new(choices, 0.0, AddressVisitor(), gen_fn, parameter_context)
    end
end

get_parameter_store(state::GFAssessState) = get_julia_store(state.parameter_context)

get_parameter_id(state::GFAssessState, name::Symbol) = (state.active_gen_fn, name)

get_active_gen_fn(state::GFAssessState) = state.active_gen_fn

function set_active_gen_fn!(state::GFAssessState, gen_fn::GenerativeFunction)
    state.active_gen_fn = gen_fn
end

function traceat(state::GFAssessState, dist::Distribution{T},
              args, key) where {T}
    local retval::T

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # get return value
    retval = get_value(state.choices, key)

    # update weight
    state.weight += logpdf(dist, retval, args...)

    return retval
end

function traceat(state::GFAssessState, gen_fn::GenerativeFunction{T,U},
              args, key) where {T,U}
    local retval::T

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # get constraints for this call
    choices = get_submap(state.choices, key)

    # get return value and weight increment
    (weight, retval) = assess(gen_fn, args, choices)

    # update score
    state.weight += weight

    return retval
end

function assess(
        gen_fn::DynamicDSLFunction, args::Tuple, choices::ChoiceMap,
        parameter_context::Dict)
    state = GFAssessState(gen_fn, choices, parameter_context)
    retval = exec(gen_fn, state, args)

    unvisited = get_unvisited(get_visited(state.visitor), choices)
    if !isempty(unvisited)
        error("Assess did not visit the following constraint addresses:\n$unvisited")
    end

    return (state.weight, retval)
end
