mutable struct GFUngenerateState
    trace::GFTrace
    constraints::Any
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
end

function GFUngenerateState(trace::GFTrace, constraints, params::Dict{Symbol,Any})
    GFUngenerateState(trace, constraints, 0., AddressVisitor(), params)
end

function addr(state::GFUngenerateState, dist::Distribution{T}, args, addr) where {T}
    visit!(state, addr)
    call::CallRecord = get_primitive_call(state.trace, addr)
    @assert call.args == args
    retval::T = call.retval
    lightweight_check_no_internal_node(state.constraints, addr)
    if has_leaf_node(state.trace, addr)
        if retval != get_leaf_node(state.trace, addr)
            error("constraints and trace do not match at $addr")
        end
        state.weight += retval.score
    end
    retval
end

function addr(state::GFUngenerateState, gen::GenerativeFunction{T}, args, addr) where {T}
    visit!(state, addr)
    subtrace = get_subtrace(state.trace, addr)
    lightweight_check_no_leaf_node(state.constraints, addr)
    if has_internal_node(state.constraints, addr)
        constraints = get_internal_node(state.constraints, addr)
    else
        constraints = EmptyAssignment()
    end
    weight = ungenerate(gen, subtrace, constraints)
    call::CallRecord = get_call_record(subtrace)
    call.retval::T
end

function splice(state::GFUngenerateState, gf::DynamicDSLFunction, args::Tuple)
    exec(gf, state, args)
end

function ungenerate(gf::DynamicDSLFunction, trace::GFTrace, constraints)
    state = GFUngenerateState(trace, constraints, gf.params)
    exec(gf, state, args)
    state.weight
end
