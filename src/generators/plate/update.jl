mutable struct PlateUpdateState{U,V,W,X}
    score::Float64
    is_empty::Bool
    args::U
    nodes::V
    changes::W
    read_trace::X
    discard::DynamicChoiceTrie
end

function update_existing_trace!(gen::Plate{T,U}, key::Int, state::PlateUpdateState, subtraces, retvals) where {T,U}
    node = haskey(state.nodes, key) ? state.nodes[key] : EmptyChoiceTrie()
    if haskey(state.changes, key)
        kernel_change = state.changes[key]
    else
        kernel_change = nothing
    end
    subtrace::U = subtraces[key]
    prev_call = get_call_record(subtrace)
    kernel_args = get_args_for_key(state.args, key)
    (subtrace, _, kernel_discard) = update(
        gen.kernel, kernel_args, kernel_change, subtrace, node, state.read_trace)
    set_internal_node!(state.discard, key, kernel_discard)
    call::CallRecord = get_call_record(subtrace)
    state.score += (call.score - prev_call.score)
    subtraces = assoc(subtraces, key, subtrace)
    retvals = assoc(retvals, key, call.retval::T)
    state.is_empty = state.is_empty && !has_choices(subtraces)
    (subtraces, retvals)
end

function assess_new_trace!(gen::Plate{T,U}, key::Int, state::PlateUpdateState, subtraces, retvals) where {T,U}
    node = haskey(state.nodes, key) ? state.nodes[key] : EmptyChoiceTrie()
    kernel_args = get_args_for_key(state.args, key)
    subtrace::U = assess(gen.kernel, kernel_args, node, state.read_trace)
    subtraces = push(subtraces, subtrace)
    call::CallRecord = get_call_record(subtrace)
    state.score += call.score
    retvals = push(retvals, call.retval::T)
    (subtraces, retvals)
end

function update_process_constraints!(nodes, to_visit, key::Int, node, new_length)
    if key <= new_length
        push!(to_visit, key)
        nodes[key] = node
    else
        error("Update did not consume constraints at key $key")
    end
end

function update_process_constraints!(nodes, to_visit, key, node, new_length)
    # key is something other than an Int
    error("Update did not consume constraints at key $key")
end

function _update(gen::Plate{T,U}, args, prev_trace::VectorTrace{T,U}, constraints, read_trace, to_visit, changes) where {T,U}

    (new_length, prev_length) = get_prev_and_new_lengths(args, prev_trace)

    # collect constraints, indexed by key
    nodes = Dict{Int, Any}()
    for (key, node) in get_internal_nodes(constraints)
        update_process_constraints!(nodes, to_visit, key, node, new_length)
    end

    # discard subtraces of deleted applications
    discard = DynamicChoiceTrie()
    for key=new_length+1:prev_length
        set_internal_node!(discard, key, get_choices(prev_trace.subtraces[key]))
    end

    # collect initial state
    state = PlateUpdateState(prev_trace.call.score, prev_trace.is_empty, args, nodes, changes, read_trace, discard)
    subtraces = prev_trace.subtraces
    retvals = prev_trace.call.retval
    
    # visit existing applications
    for key in to_visit
        (subtraces, retvals) = update_existing_trace!(gen, key, state, subtraces, retvals)
    end

    # visit each new application and assess the new trace
    for key=prev_length+1:new_length
        (subtraces, retvals) = assess_new_trace!(gen, key, state, subtraces, retvals)
    end

    call = CallRecord(state.score, retvals, args)
    trace = VectorTrace(subtraces, call, state.is_empty)

    # compute the weight
    prev_score = prev_trace.call.score
    weight = state.score - prev_score
    retchange = nothing
    (trace, weight, state.discard, retchange)
end

# No change to arguments
function update(gen::Plate{T,U}, args, change::NoChange, trace::VectorTrace{T,U}, constraints, read_trace) where {T,U}
    to_visit = Set{Int}()
    changes = Dict{Int, T}()
    _update(gen, args, trace, constraints, read_trace, to_visit, changes)
end

# Known change to arguments
function update(gen::Plate{T,U}, args, change::PlateChange{T}, trace::VectorTrace{T,U}, constraints, read_trace) where {T,U}

    (new_length, prev_length) = get_prev_and_new_lengths(args, trace)

    # calculate which existing applications to visit
    to_visit = Set{Int}()
    changes = Dict{Int, T}()
    for (i, kernel_change) in zip(change.changed_args, change.sub_changes)
        if i <= new_length
            push!(to_visit, i)
            changes[i] = kernel_change
        end
    end

    _update(gen, args, trace, constraints, read_trace, to_visit, changes)
end

# Unknown change to arguments
function update(gen::Plate{T,U}, args, change::Nothing, trace::VectorTrace{T,U}, constraints, read_trace) where {T,U}

    (new_length, prev_length) = get_prev_and_new_lengths(args, trace)

    # visit all existing applications that are preserved
    to_visit = Set(1:min(prev_length, new_length))
    changes = Dict{Int, Any}() # not used

    _update(gen, args, trace, constraints, read_trace, to_visit, changes)
end
