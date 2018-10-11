mutable struct GFRegenerateState
    prev_trace::GFTrace
    trace::GFTrace
    selection::AddressSet
    score::Float64
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
    args_change::Any
    retchange::ChangeInfo
    callee_output_changes::HomogenousTrie{Any,ChangeInfo}
end

function GFRegenerateState(args_change, prev_trace, selection, params)
    visitor = AddressVisitor()
    GFRegenerateState(prev_trace, GFTrace(), selection, 0., 0., visitor,
                    params, args_change, nothing, HomogenousTrie{Any,ChangeInfo}())
end

get_args_change(state::GFRegenerateState) = state.args_change

function set_ret_change!(state::GFRegenerateState, value)
    if state.retchange === nothing
        state.retchange = value
    else
        error("@retchange! was already used")
    end
end

function get_addr_change(state::GFRegenerateState, addr)
    get_leaf_node(state.callee_output_changes, addr)
end

function addr(state::GFRegenerateState, dist::Distribution{T}, args, addr, args_change) where {T}
    visit!(state.visitor, addr)
    has_previous = has_primitive_call(state.prev_trace, addr)
    if has_internal_node(state.selection, addr)
        error("Got internal node but expected leaf node in selection at $addr")
    end
    in_selection = has_leaf_node(state.selection, addr)
    local retval::T
    if has_previous && !in_selection
        # there was a previous value, and it was not in the selection
        # use the previous value
        # it contributes to the weight
        prev_call = get_primitive_call(state.prev_trace, addr)
        retval = prev_call.retval
        score = logpdf(dist, retval, args...)
        state.weight += score - prev_call.score
        retchange = NoChange()
    elseif has_previous
        # there was a previous value, and it was in the selection
        # simulate a new value
        # it does not contribute to the weight
        retval = random(dist, args...)
        score = logpdf(dist, retval, args...)
        prev_call = get_primitive_call(state.prev_trace, addr)
        retchange = Some(prev_call.retval)
    else
        # there is no previous value
        # simulate a new value
        # it does not contribute to the weight
        retval = random(dist, args...)
        score = logpdf(dist, retval, args...)
        retchange = nothing
    end
    set_leaf_node!(state.callee_output_changes, addr, retchange)
    call = CallRecord(score, retval, args)
    state.trace = assoc_primitive_call(state.trace, addr, call)
    state.score += score
    retval
end

function addr(state::GFRegenerateState, gen::Generator{T}, args, addr, args_change) where {T}
    visit!(state.visitor, addr)
    has_previous = has_subtrace(state.prev_trace, addr)
    if has_previous
        if addr in state.selection
            error("Entire sub-traces cannot be selected, tried to select $addr")
        end
        prev_trace = get_subtrace(state.prev_trace, addr) 
        prev_call = get_call_record(prev_trace)
        if has_internal_node(state.selection, addr)
            selection = get_internal_node(state.selection, addr)
        else
            selection = EmptyAddressSet()
        end
        (trace, weight, retchange) = regenerate(
            gen, args, args_change, prev_trace, selection)
        state.weight += weight
        set_leaf_node!(state.callee_output_changes, addr, retchange)
    else
        trace = simulate(gen, args)
        set_leaf_node!(state.callee_output_changes, addr, nothing)
    end
    call = get_call_record(trace)
    state.score += call.score
    state.trace = assoc_subtrace(state.trace, addr, trace)
    call.retval::T
end

function splice(state::GFRegenerateState, gf::GenFunction, args::Tuple)
    exec(gf, state, args)
end

function regenerate(gen::GenFunction, new_args, args_change, trace, selection)
    state = GFRegenerateState(args_change, trace, selection, gen.params)
    retval = exec(gen, state, new_args)
    new_call = CallRecord(state.score, retval, new_args)
    state.trace.call = new_call
    (state.trace, state.weight, state.retchange)
end
