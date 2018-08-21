mutable struct GFUngenerateState
    trace::GFTrace
    read_trace::Union{Some{Any},Nothing}
    constraints::Any
    weight::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
end

function GFUngenerateState(trace::GFTrace, constraints, read_trace, params::Dict{Symbol,Any})
    GFUngenerateState(trace, read_trace, constraints, 0., AddressVisitor(), params)
end

function addr(state::GFUngenerateState, dist::Distribution{T}, args, addr) where {T}
    visit!(state, addr)
    call::CallRecord = get_primitive_call(state.trace, addr)
    @assert call.args == args
    retval::T = call.retval
    if has_leaf_node(state.trace, addr)
        if retval != get_leaf_node(state.trace, addr)
            error("constraints and trace do not match at $addr")
        end
        state.weight += retval.score
    end
    retval
end

function addr(state::GFUngenerateState, gen::Generator{T}, args, addr) where {T}
    visit!(state, addr)
    subtrace = get_subtrace(state.trace, addr)
    if has_internal_node(state.constraints, addr)
        constraints = get_internal_node(state.constraints, addr)
    else
        constraints = EmptyChoiceTrie()
    end
    weight = ungenerate(gen, subtrace, constraints, state.read_trace)
    call::CallRecord = get_call_record(subtrace)
    call.retval::T
end

function splice(state::GFUngenerateState, gf::GenFunction, args::Tuple)
    exec(gf, state, args)
end

function ungenerate(gf::GenFunction, trace::GFTrace, constraints, read_trace=nothing)
    state = GFUngenerateState(trace, constraints, read_trace, gf.params)
    exec(gf, state, args)
    state.weight
end
