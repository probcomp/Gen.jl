mutable struct MapUpdateState{T,U}
    weight::Float64
    score::Float64
    noise::Float64
    subtraces::PersistentVector{U}
    retval::PersistentVector{T}
    discard::DynamicChoiceMap
    num_nonempty::Int
    updated_retdiffs::Dict{Int,Diff}
end

function get_kernel_argdiffs(argdiffs::Tuple)
    n_args = length(argdiffs)
    kernel_argdiffs = Dict{Int,Vector}()
    for (i, diff) in enumerate(argdiffs)
        if isa(diff, VectorDiff)
            for (key, element_diff) in diff.updated
                if !haskey(kernel_argdiffs, key)
                    kernel_argdiff = Vector{Any}(undef, n_args)
                    fill!(kernel_argdiff, NoChange())
                    kernel_argdiffs[key] = kernel_argdiff
                end
                kernel_argdiffs[key][i] = diff.updated[key]
            end
        end
    end
    kernel_argdiffs
end

function process_all_retained!(gen_fn::Map{T,U}, args::Tuple, argdiffs::Tuple,
                               spec::UpdateSpec, prev_length::Int, new_length::Int,
                               retained_and_targeted::Set{Int}, externally_constrained_addrs, state) where {T,U}
    kernel_no_change_argdiffs = map((_) -> NoChange(), args)
    kernel_unknown_change_argdiffs = map((_) -> UnknownChange(), args)

    if all(diff == NoChange() for diff in argdiffs)

        # only visit retained applications that were targeted
        for key in retained_and_targeted 
            @assert key <= min(new_length, prev_length)
            process_retained!(gen_fn, args, spec, key, kernel_no_change_argdiffs, externally_constrained_addrs, state)
        end

    elseif any(diff == UnknownChange() for diff in argdiffs)

        # visit every retained application
        for key in 1:min(prev_length, new_length)
            @assert key <= min(new_length, prev_length)
            process_retained!(gen_fn, args, spec, key, kernel_unknown_change_argdiffs, externally_constrained_addrs, state)
        end

    else

        key_to_kernel_argdiffs = get_kernel_argdiffs(argdiffs)

        # visit every retained applications that either has an argdiff or was targeted
        for key in union(keys(key_to_kernel_argdiffs), retained_and_targeted)
            @assert key <= min(new_length, prev_length)
            if haskey(key_to_kernel_argdiffs, key)
                kernel_argdiffs = tuple(key_to_kernel_argdiffs[key]...)
            else
                kernel_argdiffs = kernel_no_change_argdiffs
            end
            process_retained!(gen_fn, args, spec, key, kernel_argdiffs, externally_constrained_addrs, state)
        end
    
    end
end

"""
Process all new applications.
"""
function process_all_new!(gen_fn::Map{T,U}, args::Tuple, spec,
                          prev_len::Int, new_len::Int, state) where {T,U}
    for key=prev_len+1:new_len
        process_new!(gen_fn, args, spec, key, state)
    end
end

function process_retained!(gen_fn::Map{T,U}, args::Tuple,
                           spec::UpdateSpec, key::Int, kernel_argdiffs::Tuple,
                           ext_const_addrs::Selection, state::MapUpdateState{T,U}) where {T,U}
    local subtrace::U
    local prev_subtrace::U
    local retval::T

    subspec = get_subtree(spec, key)
    sub_ext_const_addrs = get_subtree(ext_const_addrs, key)
    kernel_args = get_args_for_key(args, key)

    # get new subtrace with recursive call to update()
    prev_subtrace = state.subtraces[key]
    (subtrace, weight, retdiff, discard) = update(
        prev_subtrace, kernel_args, kernel_argdiffs, subspec, sub_ext_const_addrs)

    # retrieve retdiff
    if retdiff != NoChange()
        state.updated_retdiffs[key] = retdiff
    end

    # update state
    state.weight += weight
    set_submap!(state.discard, key, discard)
    state.score += (get_score(subtrace) - get_score(prev_subtrace))
    state.noise += (project(subtrace, EmptySelection()) - project(prev_subtrace, EmptySelection()))
    state.subtraces = assoc(state.subtraces, key, subtrace)
    retval = get_retval(subtrace)
    state.retval = assoc(state.retval, key, retval)
    subtrace_empty = isempty(get_choices(subtrace))
    prev_subtrace_empty = isempty(get_choices(prev_subtrace))
    if !subtrace_empty && prev_subtrace_empty
        state.num_nonempty += 1
    elseif subtrace_empty && !prev_subtrace_empty
        state.num_nonempty -= 1
    end
end

function process_new!(gen_fn::Map{T,U}, args::Tuple, choices, key::Int,
                      state::MapUpdateState{T,U}) where {T,U}
    local subtrace::U
    local retval::T

    submap = get_subtree(choices, key)
    kernel_args = get_args_for_key(args, key)

    # get subtrace and weight
    (subtrace, weight) = generate(gen_fn.kernel, kernel_args, submap)

    # update state
    state.weight += weight
    state.score += get_score(subtrace)
    retval = get_retval(subtrace)
    @assert key > length(state.subtraces)
    state.subtraces = push(state.subtraces, subtrace)
    state.retval = push(state.retval, retval)
    @assert length(state.subtraces) == key
    if !isempty(get_choices(subtrace))
        state.num_nonempty += 1
    end
end


function update(trace::VectorTrace{MapType,T,U}, args::Tuple, argdiffs::Tuple,
                spec::UpdateSpec, externally_constrained_addrs::Selection) where {T,U}
    gen_fn = trace.gen_fn
    (new_length, prev_length) = get_prev_and_new_lengths(args, trace)
    retained_and_constrained = get_retained_and_specd(spec, prev_length, new_length)
    # TODO: for performance, don't use a Set for `retained_and_constrained`

    # handle removed applications
    (discard, num_nonempty, score_decrement, noise_decrement, weight_decrement) = vector_update_delete(
        new_length, prev_length, trace, externally_constrained_addrs)
    (subtraces, retval) = vector_remove_deleted_applications(
        trace.subtraces, trace.retval, prev_length, new_length)
    score = trace.score - score_decrement
    noise = trace.noise - noise_decrement
    
    # handle retained and new applications
    state = MapUpdateState{T,U}(-weight_decrement, score, noise,
                                     subtraces, retval, discard, num_nonempty,
                                     Dict{Int,Diff}())
    process_all_retained!(gen_fn, args, argdiffs, spec, prev_length, new_length,    
                          retained_and_constrained, externally_constrained_addrs, state)
    process_all_new!(gen_fn, args, spec, prev_length, new_length, state)

    # retdiff
    retdiff = vector_compute_retdiff(state.updated_retdiffs, new_length, prev_length)

    # new trace
    new_trace = VectorTrace{MapType,T,U}(gen_fn, state.subtraces, state.retval, args,  
        state.score, state.noise, new_length, state.num_nonempty)

    return (new_trace, state.weight, retdiff, discard)
end
