mutable struct GFUngenerateState
    trace::GFTrace
    selection::AddressSet
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
end

function GFUngenerateState(trace::GFTrace, selection, params::Dict{Symbol,Any})
    GFUngenerateState(trace, selection, 0., AddressVisitor(), params)
end

function addr(state::GFUngenerateState, dist::Distribution{T}, args, addr) where {T}
    visit!(state, addr)
    call::CallRecord = get_primitive_call(state.trace, addr)
    @assert call.args == args
    retval::T = call.retval
    lightweight_check_no_internal_node(state.selection, addr)
    if has_leaf_node(state.selection, addr)
        state.weight += retval.score
    end
    retval
end

function addr(state::GFUngenerateState, gen::GenerativeFunction{T}, args, addr) where {T}
    visit!(state, addr)
    subtrace = get_subtrace(state.trace, addr)
    lightweight_check_no_leaf_node(state.selection, addr)
    if has_internal_node(state.selection, addr)
        selection = get_internal_node(state.selection, addr)
    else
        seletion = EmptyAddressSet()
    end
    state.weight += project(gen, subtrace, selection)
    call::CallRecord = get_call_record(subtrace)
    call.retval::T
end

function splice(state::GFUngenerateState, gf::DynamicDSLFunction, args::Tuple)
    exec(gf, state, args)
end

function ungenerate(gf::DynamicDSLFunction, trace::GFTrace, selection)
    state = GFUngenerateState(trace, selection, gf.params)
    exec(gf, state, args)
    state.weight
end
