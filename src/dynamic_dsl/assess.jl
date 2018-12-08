mutable struct GFAssessState
    assmt::Assignment
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
end

function GFAssessState(constraints, params::Dict{Symbol,Any})
    GFAssessState(constraints, 0., AddressVisitor(), params)
end

function addr(state::GFAssessState, dist::Distribution{T},
              args, key) where {T}
    local retval::T

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # get return value
    retval = get_value(state.constraints, key)

    # update weight
    state.weight += logpdf(dist, retval, args...)

    retval
end

function addr(state::GFAssessState, gen_fn::GenerativeFunction{T,U},
              args, key) where {T,U}
    local retval::T

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # get constraints for this call
    assmt = get_subassmt(state.assmt, key)

    # get return value and weight increment
    (weight, retval) = assess(gen_fn, args, assmt)

    # update score
    state.weight += weight

    retval
end

function splice(state::GFAssessState, gen_fn::DynamicDSLFunction, args::Tuple)
    exec(gen_fn, state, args)
end

function assess(gen_fn::DynamicDSLFunction, args::Tuple, assmt::Assignment)
    state = Gen.GFAssessState(assmt, gen_fn.params)
    retval = exec(gen_fn, state, args) 

    if !all_visited(visited, constraints)
        error("Did not visit all constraints")
    end

    (state.weight, retval)
end
