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

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # sample return value
    retval::T = random(dist, args...)

    # update assignment
    state.assmt[key] = retval

    # update weight 
    state.weight += logpdf(dist, retval, args...)
    
    return retval
end

function addr(state::GFProposeState, gen::GenerativeFunction{T,U},
              args, key) where {T,U}

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # get subtrace
    (assmt, weight, retval::T) = propose(gen, args)

    # update assignment
    set_internal_node!(state.assmt, key, assmt)

    # update weight
    state.weight += weight

    return retval
end

splice(state::GFProposeState, gf::DynamicDSLFunction, args::Tuple) = exec(gf, state, args)

function propose(gen::DynamicDSLFunction, args::Tuple)
    state = GFProposeState(gen.params)
    retval = exec(gen, state, args)
    (state.assmt, state.weight, retval)
end
