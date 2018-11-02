"""
No change to the arguments for any retained application
"""
function process_all_retained!(gen::Plate{T,U}, args::Tuple, argdiff::NoArgDiff,
                               constraints::Dict{Int,Any},
                               prev_length::Int, new_length::Int,
                               retained_constrained::Set{Int},
                               state) where {T,U}
    # only visit retained applications that were constrained
    retained_to_visit = retained_constrained
    for key in retained_to_visit
        @assert key <= min(new_length, prev_length)
        process_retained!(gen, args, constraints, key, noargdiff, state)
    end
end

"""
Unknown change to the arguments for retained applications
"""
function process_all_retained!(gen::Plate{T,U}, args::Tuple, argdiff::UnknownArgDiff,
                               constraints::Dict{Int,Any},
                               prev_length::Int, new_length::Int,
                               retained_constrained::Set{Int},
                               state) where {T,U}
    # visit every retained application
    retained_to_visit = 1:min(prev_length, new_length)
    for key in retained_to_visit
        @assert key <= min(new_length, prev_length)
        process_retained!(gen, args, constraints, key, unknownargdiff, state)
    end
end

"""
Custom argdiffs for some retained applications
"""
function process_all_retained!(gen::Plate{T,U}, args::Tuple, argdiff::PlateCustomArgDiff{T},
                               constraints::Dict{Int,Any},
                               prev_length::Int, new_length::Int,
                               retained_constrained::Set{Int},
                               state) where {T,U}
    # visit every retained applications with a custom argdiff or constraints
    retained_to_visit = union(keys(argdiff.retained_argdiffs), retained_constrained)
    for key in retained_to_visit
        @assert key <= min(new_length, prev_length)
        if haskey(argdiff.retained_retdiffs, key)
            subargdiff = retained_retdiffs[key]
        else
            subargdiff = noargdiff
        end
        process_retained!(gen, args, constraints, key, subargdiff, state)
    end
end

"""
Process all new applications.
"""
function process_all_new!(gen::Plate{T,U}, args::Tuple, constraints::Dict{Int,Any},
                          prev_len::Int, new_len::Int,
                          state) where {T,U}
    for key=prev_len+1:new_len
        process_new!(gen, arg, constraints, key, state)
    end
end




################
# force update #
################

mutable struct PlateUpdateState{T,U}
    score::Float64
    subtraces::PersistentVector{U}
    retvals::PersistentVector{T}
    discard::DynamicAssignment
    len::Int
    num_has_choices::Int
    isdiff_retdiffs::Dict{Int,Any}
end

function process_retained!(gen::Plate{T,U}, args::Tuple,
                           constraints::Dict{Int,Any}, key::Int, kernel_argdiff,
                           state::PlateUpdateState{T,U}) where {T,U}
    
    # check for constraint
    if haskey(constraints, key)
        subconstraints = constraints[key]
    else
        subconstraints = EmptyAssignment()
    end

    # arguments for this application
    kernel_args = get_args_for_key(args, key)

    # get new subtrace with recursive call to update()
    prev_subtrace = state.subtraces[key]
    prev_call = get_call_record(prev_subtrace)
    (subtrace, _, kernel_discard, subretdiff) = update(
        gen.kernel, kernel_args, kernel_argdiff, prev_subtrace, subconstraints)
    if !isnodiff(subretdiff)
        state.isdiff_retdiffs[key] = subretdiff
    end

    # update state
    set_internal_node!(state.discard, key, kernel_discard)
    call = get_call_record(subtrace)
    state.score += (call.score - prev_call.score)
    state.subtraces = assoc(state.subtraces, key, subtrace)
    state.retvals = assoc(state.retvals, key, call.retval::T)
    if has_choices(subtrace) && !has_choices(prev_subtrace)
        state.num_has_choices += 1
    elseif !has_choices(subtrace) && has_choices(prev_subtrace)
        state.num_has_choices -= 1
    end
end

function process_new!(gen::Plate{T,U}, args::Tuple, 
                      constraints::Dict{Int,Any}, key::Int,
                      state::PlateUpdateState{T,U}) where {T,U}

    # check for constraint
    if haskey(constraints, key)
        subconstraints = constraints[key]
    else
        subconstraints = EmptyAssignment()
    end

    # extract arguments for this application
    kernel_args = get_args_for_key(args, key)

    # get subtrace
    subtrace::U = assess(gen.kernel, kernel_args, subconstraints)

    # update state
    call = get_call_record(subtrace)
    state.score += call.score
    retval::T = call.retval
    if key <= length(state.subtraces)
        state.subtraces = assoc(state.subtraces, key, subtrace)
        state.retvals = assoc(state.retvals, key, retval)
    else
        state.subtraces = push(state.subtraces, subtrace)
        state.retvals = push(state.retvals, retval)
        @assert length(state.subtraces) == key
    end
    @assert state.len == key - 1
    state.len = key
    if has_choices(subtrace)
        state.num_has_choices += 1
    end
end

function get_trace_and_weight(args::Tuple, prev_trace, state::PlateUpdateState{T,U}) where {T,U}
    call = CallRecord(state.score, state.retvals, args)
    trace = VectorTrace{T,U}(state.subtraces, state.retvals, args, state.score,
                        state.len, state.num_has_choices)
    prev_score = get_call_record(prev_trace).score
    weight = state.score - prev_score
    (trace, weight)
end

function update(gen::Plate{T,U}, args::Tuple, argdiff, prev_trace::VectorTrace{T,U},
                constraints::Assignment) where {T,U}
    (new_length, prev_length) = get_prev_and_new_lengths(args, prev_trace)
    (nodes, retained_constrained) = collect_plate_constraints(constraints, prev_length, new_length)
    (discard, num_has_choices) = discard_deleted_applications(new_length, prev_length, prev_trace)
    state = PlateUpdateState{T,U}(prev_trace.call.score,
                                  prev_trace.subtraces, prev_trace.call.retval,
                                  discard, min(prev_length, new_length), num_has_choices,
                                  Dict{Int,Any}())
    process_all_retained!(gen, args, argdiff, nodes, prev_length, new_length, retained_constrained, state)
    process_all_new!(gen, args, nodes, prev_length, new_length, state)
    (trace, weight) = get_trace_and_weight(args, prev_trace, state)
    retdiff = compute_retdiff(state.isdiff_retdiffs, new_length, prev_length)
    return (trace, weight, discard, retdiff)
end


##########
# extend #
##########

mutable struct PlateExtendState{T,U}
    score::Float64
    weight::Float64
    subtraces::PersistentVector{U}
    retvals::PersistentVector{T}
    len::Int
    num_has_choices::Bool
    isdiff_retdiffs::Dict{Int,Any}
end

function process_retained!(gen::Plate{T,U}, args::Tuple,
                           constraints::Dict{Int,Any}, key::Int, kernel_argdiff,
                           state::PlateExtendState{T,U}) where {T,U}
    
    # check for constraint
    if haskey(constraints, key)
        subconstraints = constraints[key]
    else
        subconstraints = EmptyAssignment()
    end

    # arguments for this application
    kernel_args = get_args_for_key(args, key)

    # get new subtrace with recursive call to extend()
    prev_subtrace = state.subtraces[key]
    prev_call = get_call_record(subtrace)
    (subtrace, _, subretdiff) = extend(
        gen.kernel, kernel_args, kernel_argdiff, prev_subtrace, subconstraints)
    if !isnodiff(subretdiff)
        state.isdiff_retdiffs[key] = subretdiff
    end

    # update state
    call = get_call_record(subtrace)
    state.score += (call.score - prev_call.score)
    state.subtraces = assoc(state.subtraces, key, subtrace)
    state.retvals = assoc(state.retvals, key, call.retval::T)
    if has_choices(subtrace) && !has_choices(prev_subtrace)
        state.num_has_choices += 1
    elseif !has_choices(subtrace) && has_choices(prev_subtrace)
        state.num_has_choices -= 1
    end
end

function process_new!(gen::Plate{T,U}, args::Tuple, 
                      constraints::Dict{Int,Any}, key::Int,
                      state::PlateExtendState{T,U}) where {T,U}

    # check for constraint
    if haskey(constraints, key)
        subconstraints = constraints[key]
    else
        subconstraints = EmptyAssignment()
    end

    # extract arguments for this application
    kernel_args = get_args_for_key(args, key)

    # get subtrace
    (subtrace, weight) = generate(gen.kernel, kernel_args, subconstraints)

    # update state
    call = get_call_record(subtrace)
    state.score += call.score
    state.weight += weight
    retval::T = call.retval
    if key <= length(state.subtraces)
        state.subtraces = assoc(state.subtraces, key, subtrace)
        state.retvals = assoc(state.retvals, key, retval)
    else
        state.subtraces = push(state.subtraces, subtrace)
        state.retvals = push(state.retvals, retval)
        @assert length(state.subtraces) == key
    end
    @assert state.len == key - 1
    state.len = key
    if has_choices(subtrace)
        state.num_has_choices += 1
    end
end

function extend(gen::Plate{T,U}, args::Tuple, argdiff, prev_trace::VectorTrace{T,U},
                constraints::Assignment) where {T,U}
    (new_length, prev_length) = get_prev_and_new_lengths(args, prev_trace)
    if new_length < prev_length
        error("Extend cannot remove addresses (prev length: $prev_length, new length: $new_length")
    end
    (nodes, retained_constrained) = collect_plate_constraints(constraints, prev_length, new_length)
    state = PlateExtendState{T,U}(prev_trace.call.score, 0.,
                                  prev_trace.subtraces, prev_trace.call.retval,
                                  prev_length, num_has_choices)
    process_all_retained!(gen, args, argdiff, nodes, prev_length, new_length, retained_constrained, state)
    process_all_new!(gen, args, nodes, prev_length, new_length, state)
    (trace, weight) = get_trace_and_weight(args, prev_trace, state)
    retdiff = compute_retdiff(state.isdiff_retdiffs, new_length, prev_length)
    return (trace, weight, retdiff)
end
