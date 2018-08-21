mutable struct GFProjectState
    constraints::Any
    trace::GFTrace
    read_trace::Nullable{Any}
    score::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
    discard::GenericChoiceTrie
end

function GFProjectState(constraints, read_trace, params)
    visitor = AddressVisitor()
    discard = GenericChoiceTrie()
    GFProjectState(constraints, GFTrace(), read_trace, 0., visitor, params, discard)
end

function addr(state::GFProjectState, dist::Distribution{T}, args, addr, delta) where {T}
    visit!(state.visitor, addr)
    retval::T = get_leaf_node(state.constraints, addr)
    score = logpdf(dist, retval, args...)
    call = CallRecord{T}(score, retval, args)
    state.trace = assoc_primitive_call(state.trace, addr, call)
    state.score += score
    retval 
end

function addr(state::GFProjectState, gen::Generator{T,U}, args, addr, delta) where {T,U}
    visit!(state.visitor, addr)
    if has_internal_node(state.constraints, addr)
        constraints = get_internal_node(state.constraints, addr) 
    else
        constraints = EmptyChoiceTrie()
    end
    (trace::U, discard) = project(gen, args, constraints, state.read_trace)
    call::CallRecord = get_call_record(trace)
    state.trace = assoc_subtrace(state.trace, addr, trace)
    set_internal_node!(state.discard, addr, discard)
    state.score += call.score
    call.retval::T
end

function splice(state::GFProjectState, gen::GenFunction, args::Tuple)
    exec(gf, state, args)
end

function project(gf::GenFunction, args, constraints, read_trace)
    state = GFProjectState(constraints, read_trace, gf.params)
    retval = exec(gf, state, args)
    new_call = CallRecord(state.score, retval, args)
    state.trace.call = Nullable{CallRecord}(new_call)
    unvisited_constraints = get_unvisited(state.visitor, state.constraints)
    merge!(state.discard, unvisited_constraints)
    (state.trace, state.discard)
end
