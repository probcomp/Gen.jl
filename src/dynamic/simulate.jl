mutable struct GFSimulateState
    trace::DynamicDSLTrace
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
end

function GFSimulateState(gen_fn::GenerativeFunction, args::Tuple, params)
    trace = DynamicDSLTrace(gen_fn, args)
    GFSimulateState(trace, AddressVisitor(), params)
end

function traceat(state::GFSimulateState, dist::Distribution{T},
              args, key) where {T}
    local retval::T

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    retval = random(dist, args...)

    # compute logpdf
    score = logpdf(dist, retval, args...)

    # add to the trace
    add_choice!(state.trace, key, retval, score)

    retval
end

function traceat(state::GFSimulateState, gen_fn::GenerativeFunction{T,U},
              args, key) where {T,U}
    local subtrace::U
    local retval::T

    # check key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # get subtrace
    subtrace = simulate(gen_fn, args)

    # add to the trace
    add_call!(state.trace, key, subtrace)

    # get return value
    retval = get_retval(subtrace)

    retval
end

function splice(state::GFSimulateState, gen_fn::DynamicDSLFunction,
                args::Tuple)
    prev_params = state.params
    state.params = gen_fn.params
    retval = exec(gen_fn, state, args)
    state.params = prev_params
    retval
end

function simulate(gen_fn::DynamicDSLFunction, args::Tuple)
    state = GFSimulateState(gen_fn, args, gen_fn.params)
    retval = exec(gen_fn, state, args)
    set_retval!(state.trace, retval)
    state.trace
end
