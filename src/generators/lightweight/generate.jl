############
# simulate #
############

mutable struct GFSimulateState
    trace::GFTrace
    score::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
end

function GFSimulateState(params::Dict{Symbol,Any})
    GFSimulateState(GFTrace(), 0., AddressVisitor(), params)
end

function addr(state::GFSimulateState, dist::Distribution{T}, args, key) where {T}

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # get return value and logpdf
    local retval::T
    retval = random(dist, args...)
    score = logpdf(dist, retval, args...)

    # update trace
    call = CallRecord(score, retval, args)
    state.trace = assoc_primitive_call(state.trace, key, call)

    # update score
    state.score += score
    
    return retval
end

function addr(state::GFSimulateState, gen::Generator{T,U}, args, key) where {T,U}

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # get subtrace
    local trace::U
    trace = simulate(gen, args)

    # get return value
    local retval::T
    retval = get_call_record(trace).retval

    # update trace
    state.trace = assoc_subtrace(state.trace, key, trace)

    # update score
    state.score += get_call_record(trace).score

    return retval
end

splice(state::GFSimulateState, gf::GenFunction, args::Tuple) = exec(gf, state, args)

function simulate(gen::GenFunction, args)
    state = GFSimulateState(gen.params)
    retval = exec(gen, state, args)
    # TODO add return type annotation for gen functions
    call = CallRecord{Any}(state.score, retval, args)
    state.trace.call = call
    state.trace
end

##########
# assess #
##########

mutable struct GFAssessState
    trace::GFTrace
    constraints::Any
    score::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
end

function GFAssessState(constraints, params::Dict{Symbol,Any})
    GFAssessState(GFTrace(), constraints, 0., AddressVisitor(), params)
end

function addr(state::GFAssessState, dist::Distribution{T}, args, key) where {T}

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # get return value and logpdf
    local retval::T
    retval = get_leaf_node(state.constraints, key)
    score = logpdf(dist, retval, args...)

    # update trace
    call = CallRecord(score, retval, args)
    state.trace = assoc_primitive_call(state.trace, key, call)

    # update score
    state.score += score

    return retval
end

function addr(state::GFAssessState, gen::Generator{T,U}, args, key) where {T,U}

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check if key is constrained
    lightweight_check_no_leaf_node(state.constraints, key)
    if has_internal_node(state.constraints, key)
        constraints = get_internal_node(state.constraints, key)
    else
        constraints = EmptyAssignment()
    end

    # get subtrace
    trace::U = assess(gen, args, constraints)

    # get return value
    local retval::T
    retval = get_call_record(trace).retval

    # update trace
    state.trace = assoc_subtrace(state.trace, key, trace)

    # update score
    state.score += get_call_record(trace).score

    return retval
end

splice(state::GFAssessState, gf::GenFunction, args::Tuple) = exec(gf, state, args)

function assess(gen::GenFunction, args, constraints)
    state = Gen.GFAssessState(constraints, gen.params)
    retval = Gen.exec(gen, state, args) 
    unconsumed = get_unvisited(state.visitor, constraints)
    if !isempty(unconsumed)
        error("Assess did not consume all constraints: $unconsumed")
    end
    # TODO add return type annotation for gen 
    call = Gen.CallRecord{Any}(state.score, retval, args)
    state.trace.call = call
    return state.trace
end

############
# generate #
############

mutable struct GFGenerateState
    trace::GFTrace
    constraints::Any
    score::Float64
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
end

function GFGenerateState(constraints, params::Dict{Symbol,Any})
    GFGenerateState(GFTrace(), constraints, 0., 0., AddressVisitor(), params)
end

function addr(state::GFGenerateState, dist::Distribution{T}, args, key) where {T}

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for constraints at this key
    lightweight_check_no_internal_node(state.constraints, key)
    constrained = has_leaf_node(state.constraints, key)

    # get return value and logpdf
    local retval::T
    if constrained
        retval = get_leaf_node(state.constraints, key)
    else
        retval = random(dist, args...)
    end
    score = logpdf(dist, retval, args...)

    # update trace, score, and weight
    call = CallRecord(score, retval, args)
    state.trace = assoc_primitive_call(state.trace, key, call)

    # update score
    state.score += score

    # update weight
    if constrained
        state.weight += score
    end

    return retval
end

function addr(state::GFGenerateState, gen::Generator{T,U}, args, key) where {T,U}

    # check key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for constraints at this key
    lightweight_check_no_leaf_node(state.constraints, key)
    if has_internal_node(state.constraints, key)
        constraints = get_internal_node(state.constraints, key)
    else
        constraints = EmptyAssignment()
    end

    # get subtrace
    local trace::U
    (trace, weight) = generate(gen, args, constraints)

    # get return value
    local retval::T
    retval = get_call_record(trace).retval

    # update trace
    state.trace = assoc_subtrace(state.trace, key, trace)

    # update score
    state.score += get_call_record(trace).score

    # update weight
    state.weight += weight

    return retval
end

splice(state::GFGenerateState, gf::GenFunction, args::Tuple) = exec(gf, state, args)

function generate(gen::GenFunction, args, constraints)
    state = GFGenerateState(constraints, gen.params)
    retval = exec(gen, state, args) 
    # TODO add return type annotation for gen
    call = CallRecord{Any}(state.score, retval, args)
    state.trace.call = call
    return (state.trace, state.weight)
end
