mutable struct GFProposeState
    assmt::DynamicAssignment
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
end

function GFProposeState(params::Dict{Symbol,Any})
    GFProposeState(DynamicAssignment(), 0., AddressVisitor(), params)
end

function addr(state::GFProposeState, dist::Distribution{T},
              args, key) where {T}
    local retval::T

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # sample return value
    retval = random(dist, args...)

    # update assignment
    set_value!(state.assmt, key, retval)

    # update weight 
    state.weight += logpdf(dist, retval, args...)
    
    retval
end

function addr(state::GFProposeState, gen_fn::GenerativeFunction{T,U},
              args, key) where {T,U}
    local retval::T

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # get subtrace
    (subassmt, weight, retval) = propose(gen_fn, args)

    # update assignment
    set_subassmt!(state.assmt, key, subassmt)

    # update weight
    state.weight += weight

    retval
end

function splice(state::GFProposeState, gen_fn::DynamicDSLFunction, args::Tuple)
    exec(gen_nn, state, args)
end

function propose(gen_fn::DynamicDSLFunction, args::Tuple)
    state = GFProposeState(gen_fn.params)
    retval = exec(gen_fn, state, args)
    (state.assmt, state.weight, retval)
end
