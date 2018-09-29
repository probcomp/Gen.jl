mutable struct GFFixUpdateState
    prev_trace::GFTrace
    trace::GFTrace
    constraints::Any
    score::Float64
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
    discard::DynamicAssignment
    args_change::Any
    retchange::Union{Some{Any},Nothing}
    callee_output_changes::HomogenousTrie{Any,Any}
end

function GFFixUpdateState(args_change, prev_trace, constraints, params)
    visitor = AddressVisitor()
    discard = DynamicAssignment()
    GFFixUpdateState(prev_trace, GFTrace(), constraints, 0., 0., visitor,
                     params, discard, args_change, nothing,
                     HomogenousTrie{Any,Any}())
end

get_args_change(state::GFFixUpdateState) = state.args_change

function set_ret_change!(state::GFFixUpdateState, value)
    if state.retchange === nothing
        state.retchange = value
    else
        lightweight_retchange_already_set_err()
    end
end

function get_addr_change(state::GFFixUpdateState, addr)
    get_leaf_node(state.callee_output_changes, addr)
end

function addr(state::GFFixUpdateState, dist::Distribution{T}, args, addr) where {T}
    visit!(state.visitor, addr)
    constrained = has_leaf_node(state.constraints, addr)
    has_previous = has_primitive_call(state.prev_trace, addr)
    if has_internal_node(state.constraints, addr)
        lightweight_got_internal_node_err(addr)
    end
    if constrained && !has_previous
        error("fix_update constrain a new address: $addr")
    end
    local retval::T
    if has_previous
        prev_call::CallRecord = get_primitive_call(state.prev_trace, addr)
        prev_retval::T = prev_call.retval
        if constrained
            retval = get_leaf_node(state.constraints, addr)
            set_leaf_node!(state.discard, addr, prev_retval)
        else
            retval = prev_retval
        end
    else
        retval = random(dist, args...)
    end
    score = logpdf(dist, retval, args...)
    call = CallRecord(score, retval, args)
    state.trace = assoc_primitive_call(state.trace, addr, call)
    state.score += score
    if has_previous
        state.weight += score - prev_call.score
    end
    retval 
end

function addr(state::GFFixUpdateState, gen::Generator{T}, args, addr, args_change) where {T}
    visit!(state.visitor, addr)
    if has_internal_node(state.constraints, addr)
        constraints = get_internal_node(state.constraints, addr)
    elseif has_leaf_node(state.constraints, addr)
        lightweight_got_leaf_node_err(addr)
    else
        constraints = EmptyAssignment()
    end
    local retval::T
    local trace::GFTrace
    if has_subtrace(state.prev_trace, addr)
        prev_trace = get_subtrace(state.prev_trace, addr)
        (trace, weight, discard, retchange) = fix_update(gen, args, args_change, prev_trace, constraints)
        state.weight += weight
        set_internal_node!(state.discard, addr, discard)
        set_leaf_node!(state.callee_output_changes, addr, retchange)
    else
        trace = simulate(gen, args)
        set_leaf_node!(state.callee_output_changes, addr, nothing)
    end
    call::CallRecord = get_call_record(trace)
    retval = call.retval
    state.trace = assoc_subtrace(state.trace, addr, trace)
    state.score += call.score
    retval 
end

function splice(state::GFFixUpdateState, gen::GenFunction, args::Tuple)
    exec(gf, state, args)
end

function fix_update(gf::GenFunction, args, args_change, prev_trace::GFTrace, constraints)
    state = GFFixUpdateState(args_change, prev_trace, constraints, gf.params)
    retval = exec(gf, state, args)
    new_call = CallRecord(state.score, retval, args)
    state.trace.call = new_call
    unconsumed = get_unvisited(state.visitor, constraints)
    if !isempty(unconsumed)
        error("Update did not consume all constraints")
    end
    (state.trace, state.weight, state.discard, state.retchange)
end
