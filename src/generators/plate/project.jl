mutable struct PlateProjectState{U,V}
    score::Float64
    isempty::Bool
    args::U
    nodes::V
    discard::DynamicAssignment
end

function project!(gen::Plate{T,U}, key::Int, state::PlateProjectState) where {T,U}
    node = haskey(state.nodes, key) ? state.nodes[key] : EmptyAssignment()
    kernel_args = get_args_for_key(state.args, key)
    (subtrace::U, kernel_discard) = project(gen.kernel, kernel_args, node)
    set_internal_node!(state.discard, key, kernel_discard)
    call::CallRecord{T} = get_call_record(subtrace)
    state.score += call.score
    state.is_empty = state.is_empty && !has_choices(subtrace)
    (subtrace, call.retval::T)
end

function project_process_constraints!(nodes, key::Int, node, len, discard)
    if key <= len
        nodes[key] = node
    else
        set_internal_node!(discard, key, node)
    end
end

function project_process_constraints!(nodes, key, node, len, discard)
    # key is something other than an Int
    set_internal_node!(discard, key, node)
end

function project(gen::Plate{T,U}, args, constraints) where {T,U}

    len = length(args[1])

    # collect constraints, indexed by key, discard nodes for keys not in trace
    discard = DynamicAssignment()
    nodes = Dict{Int, Any}()
    for (key, node) in get_internal_nodes(constraints)
        project_process_constraints!(nodes, key, node, len, discard)
    end

    # collect initial state
    state = PlateProjectState(gen, 0., true, args, nodes, discard)
    subtraces = Vector{U}(len)
    retvals = Vector{T}(len)
    
    # visit each new application and assess the new trace
    for key=1:len
        (subtraces[key], retvals[key]) = project!(key, state)
    end

    call = CallRecord(state.score, PersistentVector{T}(retvals), args)
    trace = VectorTrace(PersistentVector{U}(subtraces), call, state.is_empty)
    (trace, state.discard)
end
