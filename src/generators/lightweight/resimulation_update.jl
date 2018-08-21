mutable struct GFResimulationUpdateState
    prev_trace::GFTrace
    trace::GFTrace
    delta::Any
    constraints::Any
    read_trace::Union{Some{Any},Nothing}
    score::Float64
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
    discard::GenericChoiceTrie
end

function GFResimulationUpdateState(delta, prev_trace, constraints, read_trace, params)
    visitor = AddressVisitor()
    discard = GenericChoiceTrie()
    GFResimulationUpdateState(prev_trace, GFTrace(), delta, constraints, read_trace, 0., 0., visitor, params, discard)
end

function addr(state::GFResimulationUpdateState, dist::Distribution{T}, args, addr, delta) where {T}
    visit!(state.visitor, addr)
    constrained = has_leaf_node(state.constraints, addr)
    has_previous = has_primitive_call(state.prev_trace, addr)
    if has_internal_node(state.constraints, addr)
        error("Got namespace of choices for a primitive distribution at $addr")
    end
    if constrained && !has_previous
        error("resimulation_update constrain a new address: $addr")
    end
    local retval::T
    if has_previous
        prev_call::CallRecord = get_primitive_call(state.prev_trace, addr)
        prev_retval::T = prev_call.retval
        if constrained
            retval = get_leaf_node(state.constraints, addr)
        else
            retval = prev_retval
            set_leaf_node!(state.discard, addr, prev_retval)
        end
        state.weight += (score - prev_call.score)
    else
        retval = rand(dist, args...)

    end
    score = logpdf(dist, retval, args...)
    call = CallRecord(score, retval, args)
    state.trace = assoc_primitive_call(state.trace, addr, call)
    state.score += score
    retval 
end

function addr(state::GFResimulationUpdateState, gen::Generator{T}, args, addr, delta) where {T}
    visit!(state.visitor, addr)
    if has_internal_node(state.constraints, addr)
        constraints = get_internal_node(state.constraints, addr)
    elseif has_leaf_node(state.constraints, addr)
        error("Expected namespace of choices, but got single choice at $addr")
    else
        constraints = EmptyChoiceTrie()
    end
    local retval::T
    local trace::GFTrace
    if has_subtrace(state.prev_trace, addr)
        prev_trace = get_subtrace(state.prev_trace, addr)
        (trace, weight, discard) = resimulation_update(gen, args, delta, prev_trace, constraints, state.read_trace)
        state.weight += weight
        set_internal_node!(state.discard, addr, discard)
    else
        trace = simulate(gen, args, state.read_trace)
    end
    call::CallRecord = get_call_record(trace)
    retval = call.retval
    state.trace = assoc_subtrace(state.trace, addr, trace)
    state.score += call.score
    retval 
end

function splice(state::GFResimulationUpdateState, gen::GenFunction, args::Tuple)
    exec(gf, state, args)
end

function resimulation_update(gf::GenFunction, args, delta, prev_trace::GFTrace, constraints, read_trace)
    state = GFResimulationUpdateState(delta, prev_trace, constraints, read_trace, gf.params)
    retval = exec(gf, state, args)
    new_call = CallRecord(state.score, retval, args)
    state.trace.call = new_call
    prev_score = get_call(prev_trace).score
    unconsumed = get_unvisited(state.visitor, constraints)
    if !isempty(unconsumed)
        error("Update did not consume all constraints")
    end
    (state.trace, state.weight, state.discard)
end
