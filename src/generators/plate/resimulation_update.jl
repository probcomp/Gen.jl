mutable struct PlateFixUpdateState{T,U,V,W}
    kernel::T
    score::Float64
    weight::Float64
    isempty::Bool
    args::U
    nodes::V
    deltas::W
    discard::DynamicAssignment
end

function fix_update_existing_trace!(key::Int, state::PlateFixUpdateState, subtraces, retvals)
    node = haskey(state.nodes, key) ? state.nodes[key] : EmptyAssignment()
    if haskey(state.deltas, key)
        kernel_delta = state.deltas[key]
    else
        kernel_delta = nothing
    end
    subtrace = state.subtraces[key]
    prev_call = get_call_record(subtrace)
    kernel_args = get_args_for_key(state.args, key)
    (subtrace, kernel_weight, kernel_discard) = fix_update(
        state.kernel, kernel_args, kernel_delta, subtrace, node)
    set_internal_node(state.discard, kernel_discard)
    state.weight += kernel_weight
    call = get_call_record(subtrace)
    state.score += (call.score - prev_call.score)
    subtraces = assoc(trace, key, subtrace)
    retvals = assoc(retvals, key, call.retval)
    state.is_empty = state.is_empty && !has_choices(subtraces)
    (subtraces, retvals)
end

function simulate_new_trace!(key::Int, state::PlateFixUpdateState, subtraces, retvals, kernel)
    kernel_args = get_args_for_key(state.args, key)
    subtrace  = simulate(gen.kernel, kernel_args)
    subtraces = push(subtraces, subtrace)
    call = get_call_record(subtrace)
    state.score += call.score
    retvals = push(retvals, call.retval)
    (subtraces, retvals)
end


function fix_update_process_constraints!(nodes, key::Int, node, prev_length, new_length)
    if key > min(prev_length, new_length)
        error("fix_update cannot constrain new or deleted addresses; offending key: $key, prev_len: $prev_length, new_len: $new_len")
    end
    if key <= new_length
        push!(to_visit, key)
        nodes[key] = node
    else
        error("Update did not consume constraints at key $key")
    end
end

function fix_update_process_constraints!(nodes, key, node, prev_length, new_length)
    # key is something other than an Int
    error("Update did not consume constraints at key $key")
end

"""
Update with argument delta information
"""
function fix_update(gen::Plate, args, delta::PlateDelta{T}, trace::VectorTrace, constraints) where {T}

    (new_length, prev_length) = get_prev_and_new_lengths(args, trace)

    # calculate which existing applications to visit
    to_visit = Set{Int}()
    deltas = Dict{Int, T}()
    for (i, kernel_delta) in zip(delta.changed_args, delta.sub_deltas)
        if i <= new_length
            push!(to_visit, i)
            deltas[i] = kernel_delta
        end
    end

    _fix_update(gen, args, delta, trace, constraints, to_visit, deltas)
end

"""
Update without argument delta information
"""
function fix_update(gen::Plate, args, delta::Nothing, trace::VectorTrace, constraints)

    (new_length, prev_length) = get_prev_and_new_lengths(args, trace)

    # visit all existing applications that are preserved
    to_visit = 1:min(prev_length, new_length)
    deltas = Dict{Int, Any}() # not used

    _fix_update(gen, args, delta, trace, constraints, to_visit, deltas)
end

function _fix_update(gen::Plate, args, delta::Nothing, trace::VectorTrace, constraints, to_visit, deltas)

    (new_length, prev_length) = get_prev_and_new_lengths(args, trace)

    # collect constraints, indexed by key
    nodes = Dict{Int, Any}()
    for (key, node) in get_internal_nodes(constraints)
        fix_update_process_constraints!(nodes, key, node, prev_length, new_length)
    end

    # collect initial state
    discard = DynamicAssignment()
    state = PlateFixUpdateState(gen.kernel, trace.call.score, 0., trace.is_empty, args, nodes, deltas, discard)
    subtraces = trace.subtraces
    retvals = trace.call.retvals
    
    # visit existing applications
    for key in to_visit
        (subtraces, retvals) = fix_update_existing_trace!(key, state, subtraces, retvals)
    end

    # visit each new application and run simulate on it
    for key=prev_length+1:new_length
        (subtraces, retvals) = simulate_new_trace!(key, state, subtraces, retvals)
    end

    trace = VectorTrace(subtraces, CallRecord(state.score, retvals, args), state.is_empty)
    (trace, state.weight, discard)
end
