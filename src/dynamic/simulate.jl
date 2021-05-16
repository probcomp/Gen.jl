mutable struct GFSimulateState
    trace::DynamicDSLTrace
    visitor::AddressVisitor
    active_gen_fn::DynamicDSLFunction # mutated by splicing
    parameter_context::Dict{Symbol,Any}
end

function GFSimulateState(gen_fn::GenerativeFunction, args::Tuple, parameter_context)
    trace = DynamicDSLTrace(gen_fn, args, parameter_context)
    GFSimulateState(trace, AddressVisitor(), parameter_context)
end

get_parameter_store(state::GFSimulateState) = get_parameter_store(state.trace)
get_parameter_id(state::GFSimulateState, name::Symbol) = (state.active_gen_fn, name)

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
    subtrace = simulate(gen_fn, args; parameter_context=state.parameter_context)

    # add to the trace
    add_call!(state.trace, key, subtrace)

    # get return value
    retval = get_retval(subtrace)

    retval
end

function splice(state::GFSimulateState, gen_fn::DynamicDSLFunction,
                args::Tuple)
    prev_gen_fn = state.active_gen_fn
    state.active_gen_fn = gen_fn
    retval = exec(gen_fn, state, args)
    state.active_gen_fn = prev_gen_fn
    return retval
end

function simulate(
        gen_fn::DynamicDSLFunction, args::Tuple;
        parameter_context=default_parameter_context)
    state = GFSimulateState(gen_fn, args, parameter_context)
    retval = exec(gen_fn, state, args)
    set_retval!(state.trace, retval)
    state.trace
end
