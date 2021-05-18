mutable struct GFProposeState
    choices::DynamicChoiceMap
    weight::Float64
    visitor::AddressVisitor
    active_gen_fn::DynamicDSLFunction # mutated by splicing
    parameter_context::Dict

    function GFProposeState(
        gen_fn::GenerativeFunction, parameter_context)
        return new(choicemap(), 0.0, AddressVisitor(), gen_fn, parameter_context)
    end
end

get_parameter_store(state::GFProposeState) = get_julia_store(state.parameter_context)

get_parameter_id(state::GFProposeState, name::Symbol) = (state.active_gen_fn, name)

get_active_gen_fn(state::GFProposeState) = state.active_gen_fn

function set_active_gen_fn!(state::GFProposeState, gen_fn::GenerativeFunction)
    state.active_gen_fn = gen_fn
end

function traceat(state::GFProposeState, dist::Distribution{T},
              args, key) where {T}
    local retval::T

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # sample return value
    retval = random(dist, args...)

    # update assignment
    set_value!(state.choices, key, retval)

    # update weight
    state.weight += logpdf(dist, retval, args...)

    return retval
end

function traceat(state::GFProposeState, gen_fn::GenerativeFunction{T,U},
              args, key) where {T,U}
    local retval::T

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # get subtrace
    (submap, weight, retval) = propose(gen_fn, args)

    # update assignment
    set_submap!(state.choices, key, submap)

    # update weight
    state.weight += weight

    return retval
end

function propose(gen_fn::DynamicDSLFunction, args::Tuple, parameter_context::Dict)
    state = GFProposeState(gen_fn, parameter_context)
    retval = exec(gen_fn, state, args)
    return (state.choices, state.weight, retval)
end
