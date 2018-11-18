mutable struct GFProjectState
    constraints::Any
    trace::GFTrace
    score::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
    discard::DynamicAssignment
end

function GFProjectState(constraints, params)
    visitor = AddressVisitor()
    discard = DynamicAssignment()
    GFProjectState(constraints, GFTrace(), 0., visitor, params, discard)
end

function addr(state::GFProjectState, dist::Distribution{T}, args, addr) where {T}
    visit!(state.visitor, addr)
    retval::T = get_leaf_node(state.constraints, addr)
    score = logpdf(dist, retval, args...)
    call = CallRecord{T}(score, retval, args)
    state.trace = assoc_primitive_call(state.trace, addr, call)
    state.score += score
    retval 
end

function addr(state::GFProjectState, gen::GenerativeFunction{T,U}, args, addr, args_change) where {T,U}
    visit!(state.visitor, addr)
    if has_internal_node(state.constraints, addr)
        constraints = get_internal_node(state.constraints, addr) 
    elseif has_leaf_node(state.constraints, addr)
        lightweight_got_leaf_node_err(addr)
    else
        constraints = EmptyAssignment()
    end
    (trace::U, discard) = project(gen, args, constraints)
    call::CallRecord = get_call_record(trace)
    state.trace = assoc_subtrace(state.trace, addr, trace)
    set_internal_node!(state.discard, addr, discard)
    state.score += call.score
    call.retval::T
end

function splice(state::GFProjectState, gen::DynamicDSLFunction, args::Tuple)
    exec(gf, state, args)
end

function project(gf::DynamicDSLFunction, args, constraints)
    state = GFProjectState(constraints, gf.params)
    retval = exec(gf, state, args)
    new_call = CallRecord(state.score, retval, args)
    state.trace.call = new_call
    unvisited_constraints = get_unvisited(state.visitor, state.constraints)
    merge!(state.discard, unvisited_constraints)
    (state.trace, state.discard)
end
