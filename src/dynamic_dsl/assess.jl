mutable struct GFAssessState
    assmt::Assignment
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
end

function GFAssessState(assmt, params::Dict{Symbol,Any})
    GFAssessState(assmt, 0., AddressVisitor(), params)
end

function addr(state::GFAssessState, dist::Distribution{T},
              args, key) where {T}
    local retval::T

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # get return value
    retval = get_value(state.assmt, key)

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
    prev_params = state.params
    state.params = gen_fn.params
    retval = exec(gen_fn, state, args)
    state.params = prev_params
    retval
end

function assess(gen_fn::DynamicDSLFunction, args::Tuple, assmt::Assignment)
    state = GFAssessState(assmt, gen_fn.params)
    retval = exec(gen_fn, state, args)

    unvisited = get_unvisited(get_visited(state.visitor), assmt)
    if !isempty(unvisited)
        error("Assess did not visit the following constraint addresses:\n$unvisited")
    end

    (state.weight, retval)
end
