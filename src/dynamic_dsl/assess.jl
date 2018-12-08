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

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # get return value
    local retval::T
    retval = get_value(state.constraints, key)

    # update weight
    state.weight += logpdf(dist, retval, args...)

    return retval
end

function addr(state::GFAssessState, gen::GenerativeFunction{T,U}, args, key) where {T,U}

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check if key is constrained
    # lightweight_check_no_leaf_node(state.assmt, key) not needed since get_subassmt crashes now if there is a value
    assmt = get_subassmt(state.assmt, key)

    # get return value and weight increment
    local retval::T
    (weight, retval) = assess(gen, args, assmt)

    # update score
    state.weight += weight

    return retval
end

splice(state::GFAssessState, gf::DynamicDSLFunction, args::Tuple) = exec(gf, state, args)

function assess(gen::DynamicDSLFunction, args::Tuple, assmt::Assignment)
    state = Gen.GFAssessState(assmt, gen.params)
    retval = Gen.exec(gen, state, args) 

    if !all_visited(visited, constraints)
        error("Did not visit all constraints")
    end

    (state.weight, retval)
end
