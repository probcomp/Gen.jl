mutable struct PlateExtendState{U,V,W}
    score::Float64
    weight::Float64
    isempty::Bool
    args::U
    nodes::V
    changes::W
end

function extend_existing_trace!(gen::Plate{T,U}, key::Int, state::PlateExtendState, subtraces, retvals) where {T,U}
    node = haskey(state.nodes, key) ? state.nodes[key] : EmptyChoiceTrie()
    if haskey(state.changes, key)
        kernel_change = state.changes[key]
    else
        kernel_change = nothing
    end
    subtrace::U = state.subtraces[key]
    prev_call = get_call_record(subtrace)
    kernel_args = get_args_for_key(state.args, key)
    (subtrace, kernel_weight) = extend(gen.kernel, kernel_args, kernel_change, subtrace, node)
    state.weight += kernel_weight
    call::CallRecord{T} = get_call_record(subtrace)
    state.score += (call.score - prev_call.score)
    subtraces = assoc(trace, key, subtrace)
    retvals = assoc(retvals, key, call.retval::T)
    state.is_empty = state.is_empty && !has_choices(subtraces)
    (subtraces, retvals)
end

function generate_new_trace!(gen::Plate{T,U}, key::Int, state::PlateExtendState, subtraces, retvals, kernel) where {T,U}
    node = haskey(state.nodes, key) ? state.nodes[key] : EmptyChoiceTrie()
    kernel_args = get_args_for_key(state.args, key)
    (subtrace::U, kernel_weight) = generate(gen.kernel, kernel_args, node)
    state.weight += kernel_weight
    subtraces = push(subtraces, subtrace)
    call::CallRecord{T} = get_call_record(subtrace)
    state.score += call.score
    retvals = push(retvals, call.retval::T)
    (subtraces, retvals)
end

function extend_process_constraints!(nodes, key::Int, node, new_length, to_visit)
    if key <= new_length
        push!(to_visit, key)
        nodes[key] = node
    else
        error("Constraints under key $key not part of trace")
    end
end

function extend_process_constraints!(nodes, key, node, new_length, to_visit)
    error("Constraints under key $key not part of trace")
end


"""
Extend with argument change information
"""
function extend(gen::Plate, args, change::PlateChange{T}, trace::VectorTrace, constraints) where {T}

    (new_length, prev_length) = get_prev_and_new_lengths(args, trace)
    if prev_length < new_length
        error("extend cannot remove random choices")
    end

    # calculate which existing applications to visit
    to_visit = Set{Int}()
    changes = Dict{Int, T}()
    for (i, kernel_change) in zip(change.changed_args, change.sub_changes)
        push!(to_visit, i)
        changes[i] = kernel_change
    end

    _extend(gen, args, trace, constraints, to_visit, changes)
end

"""
Extend without argument change information
"""
function extend(gen::Plate{T,U}, args, change::NoChange, trace::VectorTrace{T,U}, constraints) where {T,U}

    (new_length, prev_length) = get_prev_and_new_lengths(args, trace)
    
    # visit all existing applications
    to_visit = 1:new_length
    changes = Dict{Int, Any}() # not used

    _extend(gen, args, trace, constraints, to_visit, changes)
end


function _extend(gen::Plate, args, trace::VectorTrace, constraints, to_visit, changes)

    (new_length, prev_length) = get_prev_and_new_lengths(args, trace)
    if prev_length < new_length
        error("extend cannot remove random choices")
    end

    # collect constraints, indexed by key, and error if any are not in the new trace
    nodes = Dict{Int, Any}()
    for (key, node) in get_internal_nodes(constraints)
        extend_process_constraints!(nodes, key, node, new_length, to_visit)
    end

    # collect initial state
    state = PlateExtendState(trace.call.score, 0., trace.is_empty, args, nodes, changes)
    subtraces = trace.subtraces
    retvals = trace.call.retvals
    
    # visit all existing applications
    for key in to_visit
        (subtraces, retvals) = extend_existing_trace!(gen, key, state, subtraces, retvals)
    end

    # visit each new application and run generate on it
    for key=prev_length+1:new_length
        (subtraces, retvals) = generate_new_trace!(gen, key, state, subtraces, retvals)
    end

    call = CallRecord(state.score, retvals, args)
    trace = VectorTrace(subtraces, call, state.is_empty)
    retchange = nothing
    (trace, state.weight, retchange)
end
