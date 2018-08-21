mutable struct GFExtendState
    prev_trace::GFTrace
    trace::GFTrace
    delta::Any
    constraints::Any
    read_trace::Union{Some{Any},Nothing}
    score::Float64
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
end

function GFExtendState(delta, prev_trace, constraints, read_trace, params)
    visitor = AddressVisitor()
    GFExtendState(prev_trace, GFTrace(), delta, constraints, read_trace, 0., 0., visitor, params)
end

function addr(state::GFExtendState, gen::Distribution{T}, args, addr, delta) where {T}
    visit!(state.visitor, addr)
    if has_leaf_node(state.constraints, addr)
        error("Cannot change value of random choice at $addr")
    end
    if has_internal_node(state.constraints, addr)
        error("Got namespace of choices for a primitive distribution at $addr")
    end
    local retval::T
    local score::Float64
    local call::CallRecord
    if has_primitive_call(state.prev_trace, addr)
        call = get_primitive_call(state.prev_trace, addr)
        if call.prev_args != args
            error("Cannot change arguments to a random choice in extend")
        end
        retval = call.retval
        score = call.score
    else
        retval = rand(dist, args...)
        score = logpdf(dist, args, retval...)
        state.weight += score
        call = CallRecord(score, retval, args)
    end
    state.trace = assoc_primitive_call(state.trace, addr, call)
    state.score += score
    trace.retval 
end

function addr(state::GFExtendState, gen::Generator{T}, args, addr, delta) where {T}
    visit!(state.visitor, addr)
    if has_internal_node(state.constraints, addr)
        constraints = get_internal_node(state.constraints, addr)
    elseif has_leaf_node(state.constraints, addr)
        error("Expected namespace of choices, but got single choice at $addr")
    else
        constraints = EmptyChoiceTrie()
    end
    if has_subtrace(state.prev_trace, addr)
        prev_trace = get_subtrace(state.prev_trace, addr)
        (trace, weight) = extend(gen, args, delta, prev_trace, constraints, state.read_trace)
    else
        (trace, weight) = generate(gen, args, constraints, state.read_trace)
    end
    call::CallRecord = get_call_record(trace)
    retval::T = call.retval
    state.trace = assoc_composite_call(state.trace, addr, trace)
    state.score += trace.score
    state.weight += weight
    retval 
end

function splice(state::GFExtendState, gen::GenFunction, args::Tuple)
    exec(gf, state, args)
end

function extend(gf::GenFunction, args, delta, trace::GFTrace, constraints, read_trace=nothing)
    state = GFExtendState(delta, trace, constraints, read_trace, gf.params)
    retval = exec(gf, state, args)
    call = CallRecord(state.score, retval, args)
    state.trace.call = call
    (state.trace, state.weight)
end
