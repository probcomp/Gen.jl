mutable struct GFExtendState
    prev_trace::GFTrace
    trace::GFTrace
    args_change::Any
    constraints::Any
    score::Float64
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
end

function GFExtendState(args_change, prev_trace, constraints, params)
    visitor = AddressVisitor()
    GFExtendState(prev_trace, GFTrace(), args_change, constraints, 0., 0., visitor, params)
end

function addr(state::GFExtendState, dist::Distribution{T}, args, addr) where {T}
    visit!(state.visitor, addr)
    has_previous = has_primitive_call(state.prev_trace, addr)
    constrained = has_leaf_node(state.constraints, addr)
    if has_previous && constrained
        error("Cannot change value of random choice at $addr")
    end
    if has_internal_node(state.constraints, addr)
        error("Got namespace of choices for a primitive distribution at $addr")
    end
    local retval::T
    local call::CallRecord
    if has_previous
        call = get_primitive_call(state.prev_trace, addr)
        if call.args != args
            error("Cannot change arguments to a random choice in extend")
        end
        retval = call.retval
        score = call.score
    elseif constrained
        retval = get_leaf_node(state.constraints, addr)
        score = logpdf(dist, retval, args...)
        state.weight += score
        call = CallRecord(score, retval, args)
    else
        retval = random(dist, args...)
        score = logpdf(dist, retval, args...)
        call = CallRecord(score, retval, args)
    end
    state.trace = assoc_primitive_call(state.trace, addr, call)
    state.score += score
    retval 
end

function addr(state::GFExtendState, gen::Generator{T}, args, addr, args_change) where {T}
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
        (trace, weight) = extend(gen, args, args_change, prev_trace, constraints)
    else
        (trace, weight) = generate(gen, args, constraints)
    end
    call::CallRecord = get_call_record(trace)
    retval::T = call.retval
    state.trace = assoc_subtrace(state.trace, addr, trace)
    state.score += call.score
    state.weight += weight
    retval 
end

function splice(state::GFExtendState, gen::GenFunction, args::Tuple)
    exec(gf, state, args)
end

function extend(gf::GenFunction, args, args_change, trace::GFTrace, constraints)
    state = GFExtendState(args_change, trace, constraints, gf.params)
    retval = exec(gf, state, args)
    call = CallRecord(state.score, retval, args)
    state.trace.call = call
    (state.trace, state.weight)
end
