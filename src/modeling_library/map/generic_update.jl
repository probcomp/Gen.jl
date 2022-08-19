
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
                               choices_or_selection, prev_length::Int, new_length::Int,
                               retained_and_targeted::Set{Int}, state) where {T,U}
    kernel_no_change_argdiffs = map((_) -> NoChange(), args)
    kernel_unknown_change_argdiffs = map((_) -> UnknownChange(), args)

    if all(diff == NoChange() for diff in argdiffs)

        # only visit retained applications that were targeted
        for key in retained_and_targeted
            @assert key <= min(new_length, prev_length)
            process_retained!(gen_fn, args, choices_or_selection, key, kernel_no_change_argdiffs, state)
        end

    elseif any(diff == UnknownChange() for diff in argdiffs)

        # visit every retained application
        for key in 1:min(prev_length, new_length)
            @assert key <= min(new_length, prev_length)
            process_retained!(gen_fn, args, choices_or_selection, key, kernel_unknown_change_argdiffs, state)
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
            process_retained!(gen_fn, args, choices_or_selection, key, kernel_argdiffs, state)
        end

    end
end

"""
Process all new applications.
"""
function process_all_new!(gen_fn::Map{T,U}, args::Tuple, choices_or_selection,
                          prev_len::Int, new_len::Int, state) where {T,U}
    for key=prev_len+1:new_len
        process_new!(gen_fn, args, choices_or_selection, key, state)
    end
end
