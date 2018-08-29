mutable struct GFSimulateState
    trace::GFTrace
    score::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
end

function GFSimulateState(params::Dict{Symbol,Any})
    GFSimulateState(GFTrace(), 0., AddressVisitor(), params)
end

get_args_change(state::GFSimulateState) = nothing
get_addr_change(state::GFSimulateState, addr) = nothing
set_ret_change!(state::GFSimulateState, value) = begin end

function addr(state::GFSimulateState, dist::Distribution{T}, args, addr, arg_change) where {T}
    visit!(state.visitor, addr)
    retval::T = random(dist, args...)
    score = logpdf(dist, retval, args...)
    call = CallRecord(score, retval, args)
    state.trace = assoc_primitive_call(state.trace, addr, call)
    state.trace.has_choices = true
    state.score += score
    retval
end

function addr(state::GFSimulateState, gen::Generator{T,U}, args, addr, arg_change) where {T,U}
    visit!(state.visitor, addr)
    trace::U = simulate(gen, args)
    call::CallRecord = get_call_record(trace)
    state.trace = assoc_subtrace(state.trace, addr, trace)
    state.trace.has_choices = state.trace.has_choices || has_choices(trace)
    state.score += call.score
    call.retval::T
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
