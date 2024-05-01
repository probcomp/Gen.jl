mutable struct GFProposeState{R<:AbstractRNG}
    choices::DynamicChoiceMap
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
    rng::R
end

function GFProposeState(params::Dict{Symbol,Any}, rng::AbstractRNG)
    GFProposeState(choicemap(), 0., AddressVisitor(), params, rng)
end

function traceat(state::GFProposeState, dist::Distribution{T},
              args, key) where {T}
    local retval::T

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # sample return value
    retval = random(state.rng, dist, args...)

    # update assignment
    set_value!(state.choices, key, retval)

    # update weight
    state.weight += logpdf(dist, retval, args...)

    retval
end

function traceat(state::GFProposeState, gen_fn::GenerativeFunction{T,U},
              args, key) where {T,U}
    local retval::T

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # get subtrace
    (submap, weight, retval) = propose(state.rng, gen_fn, args)

    # update assignment
    set_submap!(state.choices, key, submap)

    # update weight
    state.weight += weight

    retval
end

function splice(state::GFProposeState, gen_fn::DynamicDSLFunction, args::Tuple)
    prev_params = state.params
    state.params = gen_fn.params
    retval = exec(gen_fn, state, args)
    state.params = prev_params
    retval
end

function propose(rng::AbstractRNG, gen_fn::DynamicDSLFunction, args::Tuple)
    state = GFProposeState(gen_fn.params, rng)
    retval = exec(gen_fn, state, args)
    (state.choices, state.weight, retval)
end
