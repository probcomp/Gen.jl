mutable struct PlateGenerateState{U,V,X}
    score::Float64
    weight::Float64
    is_empty::Bool
    args::U
    nodes::V
    read_trace::X
end


function generate_new_trace!(gen::Plate{T,U}, key::Int, state::PlateGenerateState) where {T,U}
    kernel_args = get_args_for_key(state.args, key)
    local subtrace::U
    if haskey(state.nodes, key)
        node = state.nodes[key]
        (subtrace, kernel_weight) = generate(gen.kernel, kernel_args, node, state.read_trace)
        state.weight += kernel_weight
    else
        subtrace = simulate(gen.kernel, kernel_args, state.read_trace)
    end
    state.is_empty = state.is_empty && !has_choices(subtrace)
    call::CallRecord = get_call_record(subtrace)
    state.score += call.score
    (subtrace, call.retval::T)
end

function generate_process_constraints!(nodes, key::Int, node, len)
    if key <= len
        nodes[key] = node
    else
        error("Constraints under key $key not part of trace")
    end
end

function generate_process_constraints!(nodes, key, node, len)
    error("Constraints under key $key not part of trace")
end

function generate(gen::Plate{T,U}, args, constraints, read_trace=nothing) where {T,U}
    len = length(args[1])
    
    # collect constraints, indexed by key, and error if any are not in the trace
    nodes = Dict{Int, Any}()
    for (key, node) in get_internal_nodes(constraints)
        generate_process_constraints!(nodes, key, node, len)
    end

    # collect initial state
    state = PlateGenerateState(0., 0., true, args, nodes, read_trace)
    subtraces = Vector{U}(undef, len)
    retvals = Vector{T}(undef, len)
    
    # visit each new application and run generate (or simulate) on it
    for key=1:len
        (subtraces[key], retvals[key]) = generate_new_trace!(gen, key, state)
    end

    call = CallRecord(state.score, PersistentVector{T}(retvals), args)
    trace = VectorTrace{T,U}(PersistentVector{U}(subtraces), call, state.is_empty)
    (trace, state.weight)
end
