mutable struct PlateAssessState{U,V}
    score::Float64
    is_empty::Bool
    args::U
    nodes::V
end

function assess_kernel!(gen::Plate{T,U}, key::Int, state::PlateAssessState) where {T,U}
    kernel_args = get_args_for_key(state.args, key)
    node = state.nodes[key]
    subtrace::U = assess(gen.kernel, kernel_args, node)
    state.is_empty = state.is_empty && !has_choices(subtrace)
    call::CallRecord{T} = get_call_record(subtrace)
    state.score += call.score
    (subtrace, call.retval::T)
end

function check_constraint_key!(key, len::Int)
    error("Constraints under key $key not part of trace")
end

function check_constraint_key!(key::Int, len::Int)
    if key < 1 || key > len
        error("Constraints under key $key not part of trace")
    end
end

function assess(gen::Plate{T,U}, args, constraints) where {T,U}
    len = length(args[1])
    
    # collect constraints, indexed by key
    nodes = Vector{Any}(len)
    for key=1:len
        nodes[key] = get_internal_node(constraints, key)
    end

    # check no other constraints.. (can be skipped if we're trying to be less strict)
    for (key, _) in get_internal_nodes(constraints)
        check_constraint!(key, len)
    end
    for (key, _) in get_leaf_nodes(constraints)
        check_constraint!(key, len)
    end

    # collect initial state
    state = PlateAssessState(0., true, args, nodes)
    subtraces = Vector{U}(len)
    retvals = Vector{T}(len)
    
    # visit each application and assess it
    for key=1:len
        (subtraces[key], retvals[key]) = assess_kernel!(gen, key, state)
    end

    call = CallRecord{PersistentVector{T}}(state.score, PersistentVector{T}(retvals), args, UnknownChange())
    VectorTrace{T,U}(PersistentVector{U}(subtraces), call, state.is_empty)
end
