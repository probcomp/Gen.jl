mutable struct GFRegenerateState
    prev_trace::GFTrace
    trace::GFTrace
    selection::AddressSet
    read_trace::Nullable{Any}
    score::Float64
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
    args_change::Any
    retchange::Nullable{Any}
    callee_output_changes::HomogenousTrie{Any,Any}
end

function GFRegenerateState(args_change, prev_trace, selection, read_trace, params)
    visitor = AddressVisitor()
    GFRegenerateState(prev_trace, GFTrace(), selection, read_trace, 0., 0., visitor,
                    params, args_change, Nullable{Any}(), HomogenousTrie{Any,Any}())
end

get_args_change(state::GFRegenerateState) = state.args_change

function set_ret_change!(state::GFRegenerateState, value)
    if isnull(state.retchange)
        state.retchange = Nullable{Any}(value)
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
    in_selection = addr in state.selection
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
        # retchange indicates that there was a change and gives the previous value
        retval = rand(dist, args...)
        score = logpdf(dist, retval, args...)
        prev_call = get_primitive_call(state.prev_trace, addr)
        retchange = (true, prev_call.retval)
    else
        # there is no previous value
        # simulate a new value
        # it does not contribute to the weight
        # retchange is nothing, because the address is new
        retval = rand(dist, args...)
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
        selection = state.selection[addr]
        (trace, weight, retchange) = regenerate(
            gen, args, args_change, prev_trace, selection, state.read_trace)
        state.weight += weight
        set_leaf_node!(state.callee_output_changes, addr, retchange)
    else
        trace = simulate(gen, args, state.read_trace)
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

function regenerate(gen::GenFunction, new_args, args_change, trace, selection, read_trace=nothing)
    state = GFRegenerateState(args_change, trace, selection, read_trace, gen.params)
    retval = exec(gen, state, new_args)
    new_call = CallRecord(state.score, retval, new_args)
    state.trace.call = Nullable{CallRecord}(new_call)
    retchange = isnull(state.retchange) ? nothing : get(state.retchange)
    (state.trace, state.weight, retchange)
end
