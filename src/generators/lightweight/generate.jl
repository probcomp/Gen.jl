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

get_args_change(state::GFGenerateState) = nothing
get_addr_change(state::GFGenerateState, addr) = nothing
set_ret_change!(state::GFGenerateState, value) = begin end

function addr(state::GFGenerateState, dist::Distribution{T}, args, addr, delta) where {T}
    visit!(state.visitor, addr)
    constrained = has_leaf_node(state.constraints, addr)
    if has_internal_node(state.constraints, addr)
        error("Got namespace of choices for a primitive distribution at $addr")
    end
    local retval::T
    if constrained
        retval = get_leaf_node(state.constraints, addr)
    else
        retval = random(dist, args...)
    end
    score = logpdf(dist, retval, args...)
    call = CallRecord(score, retval, args)
    state.trace = assoc_primitive_call(state.trace, addr, call)
    state.trace.has_choices = true
    state.score += score
    if constrained
        state.weight += score
    end
    retval
end

function addr(state::GFGenerateState, gen::Generator{T,U}, args, addr, delta) where {T,U}
    visit!(state.visitor, addr)
    if has_internal_node(state.constraints, addr)
        constraints = get_internal_node(state.constraints, addr)
    else
        constraints = EmptyChoiceTrie()
    end
    (trace::U, weight) = generate(gen, args, constraints)
    call::CallRecord = get_call_record(trace)
    state.trace = assoc_subtrace(state.trace, addr, trace)
    state.trace.has_choices |= has_choices(trace)
    state.score += call.score
    state.weight += weight
    call.retval::T
end

splice(state::GFGenerateState, gf::GenFunction, args::Tuple) = exec(gf, state, args)

function generate(gen::GenFunction, args, constraints)
    state = GFGenerateState(constraints, gen.params)
    retval = exec(gen, state, args) 
    # TODO add return type annotation for gen
    call = CallRecord{Any}(state.score, retval, args)
    state.trace.call = call
    (state.trace, state.weight)
end
