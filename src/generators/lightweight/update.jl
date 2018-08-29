mutable struct GFUpdateState
    prev_trace::GFTrace
    trace::GFTrace
    constraints::Any
    score::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
    discard::DynamicChoiceTrie
    args_change::Any
    retchange::Union{Some{Any},Nothing}
    callee_output_changes::HomogenousTrie{Any,Any}
end

function GFUpdateState(args_change, prev_trace, constraints, params)
    visitor = AddressVisitor()
    discard = DynamicChoiceTrie()
    GFUpdateState(prev_trace, GFTrace(), constraints, 0., visitor,
                  params, discard, args_change, nothing, HomogenousTrie{Any,Any}())
end

get_args_change(state::GFUpdateState) = state.args_change

function set_ret_change!(state::GFUpdateState, value)
    if state.retchange === nothing
        state.retchange = value
    else
        error("@retchange! was already used")
    end
end

function get_addr_change(state::GFUpdateState, addr)
    get_leaf_node(state.callee_output_changes, addr)
end

function addr(state::GFUpdateState, dist::Distribution{T}, args, addr) where {T}
    visit!(state.visitor, addr)
    constrained = has_leaf_node(state.constraints, addr)
    has_previous = has_primitive_call(state.prev_trace, addr)
    if has_internal_node(state.constraints, addr)
        error("Got namespace of choices for a primitive distribution at $addr")
    end
    local retval::T
    if has_previous
        prev_call::CallRecord = get_primitive_call(state.prev_trace, addr)
    end
    if constrained
        retval = get_leaf_node(state.constraints, addr)
    elseif has_previous
        retval = prev_call.retval
    else
        error("Constraint not given for new address: $addr")
    end
    if constrained && has_previous
        # there was a change and this was the previous value
        retchange = (true, prev_call.retval)
    elseif has_previous
        retchange = NoChange()
    else
        # retchange is null, because the address is new
        retchange = nothing
    end
    score = logpdf(dist, retval, args...)
    call = CallRecord(score, retval, args)
    state.trace = assoc_primitive_call(state.trace, addr, call)
    state.score += score
    if constrained && has_previous
        set_leaf_node!(state.discard, addr, prev_call.retval)
    end
    set_leaf_node!(state.callee_output_changes, addr, retchange)
    retval 
end

function addr(state::GFUpdateState, gen::Generator{T}, args, addr, args_change) where {T}
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
        (trace, _, discard, retchange) = update(gen, args, args_change,
            prev_trace, constraints)
        set_internal_node!(state.discard, addr, discard)
        set_leaf_node!(state.callee_output_changes, addr, retchange)
    else
        trace = assess(gen, args, constraints)
        set_leaf_node!(state.callee_output_changes, addr, NoChange())
    end
    call::CallRecord = get_call_record(trace)
    retval::T = call.retval
    state.trace = assoc_subtrace(state.trace, addr, trace)
    state.score += call.score
    retval 
end

splice(state::GFUpdateState, gen::GenFunction, args::Tuple) = exec(gf, state, args)

function update(gen::GenFunction, new_args, args_change, trace::GFTrace, constraints)
    state = Gen.GFUpdateState(args_change, trace, constraints, gen.params)
    retval = Gen.exec(gen, state, new_args)
    new_call = Gen.CallRecord{Any}(state.score, retval, new_args)
    state.trace.call = new_call
    # discard addresses that were deleted
    unvisited = Gen.get_unvisited(state.visitor, get_choices(state.prev_trace))
    merge!(state.discard, unvisited) # TODO
    if !isempty(Gen.get_unvisited(state.visitor, constraints))
        error("Update did not consume all constraints")
    end
    
    # compute the weight
    prev_score = get_call_record(trace).score
    weight = state.score - prev_score
    (state.trace, weight, state.discard, state.retchange)
end
