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

get_args_change(state::GFAssessState) = nothing
get_addr_change(state::GFAssessState, addr) = nothing
set_ret_change!(state::GFAssessState, value) = begin end

function addr(state::GFAssessState, dist::Distribution{T}, args, addr, delta) where {T}
    visit!(state.visitor, addr)
    retval::T = get_leaf_node(state.constraints, addr)
    score = logpdf(dist, retval, args...)
    call = CallRecord(score, retval, args)
    state.trace = assoc_primitive_call(state.trace, addr, call)
    state.trace.has_choices = true
    state.score += score
    retval
end

function addr(state::GFAssessState, gen::Generator{T,U}, args, addr, delta) where {T,U}
    visit!(state.visitor, addr)
    if has_internal_node(state.constraints, addr)
        constraints = get_internal_node(state.constraints, addr)
    else
        constraints = EmptyChoiceTrie()
    end
    trace::U = assess(gen, args, constraints)
    call::CallRecord = get_call_record(trace)
    state.trace = assoc_subtrace(state.trace, addr, trace)
    state.trace.has_choices |= has_choices(trace)
    state.score += call.score
    call.retval::T
end

splice(state::GFAssessState, gf::GenFunction, args::Tuple) = exec(gf, state, args)

function assess(gen::GenFunction, args, constraints)
    state = Gen.GFAssessState(constraints, gen.params)
    retval = Gen.exec(gen, state, args) 
    # TODO check that there were no unvisited constraints
    unconsumed = get_unvisited(state.visitor, constraints)
    if !isempty(unconsumed)
        error("Update did not consume all constraints: $unconsumed")
    end

    # TODO add return type annotation for gen 
    call = Gen.CallRecord{Any}(state.score, retval, args)
    state.trace.call = call
    state.trace
end
