"""
No change to the arguments for any retained application
"""
function process_all_retained!(gen_fn::Map{T,U}, args::Tuple, ::NoArgDiff,
                               assmt_or_selection, prev_length::Int, new_length::Int,
                               retained_and_targeted::Set{Int}, state) where {T,U}

    # only visit retained applications that were targeted
    for key in retained_and_targeted 
        @assert key <= min(new_length, prev_length)
        process_retained!(gen_fn, args, assmt_or_selection, key, noargdiff, state)
    end
end

"""
Unknown change to the arguments for retained applications
"""
function process_all_retained!(gen_fn::Map{T,U}, args::Tuple, ::UnknownArgDiff,
                               assmt_or_selection, prev_length::Int, new_length::Int,
                               ::Set{Int}, state) where {T,U}

    # visit every retained application
    for key in 1:min(prev_length, new_length)
        @assert key <= min(new_length, prev_length)
        process_retained!(gen_fn, args, assmt_or_selection, key, unknownargdiff, state)
    end
end

"""
Custom argdiffs for some retained applications
"""
function process_all_retained!(gen_fn::Map{T,U}, args::Tuple, argdiff::MapCustomArgDiff,
                               assmt_or_selection, prev_length::Int, new_length::Int,
                               retained_and_targeted::Set{Int}, state) where {T,U}

    # visit every retained applications with an argdiff or that was targeted
    for key in union(keys(argdiff.retained_argdiffs), retained_and_targeted)
        @assert key <= min(new_length, prev_length)
        if haskey(argdiff.retained_argdiffs, key)
            subargdiff = argdiff.retained_argdiffs[key]
        else
            subargdiff = noargdiff
        end
        process_retained!(gen_fn, args, assmt_or_selection, key, subargdiff, state)
    end
end

"""
Process all new applications.
"""
function process_all_new!(gen_fn::Map{T,U}, args::Tuple, assmt_or_selection,
                          prev_len::Int, new_len::Int, state) where {T,U}
    for key=prev_len+1:new_len
        process_new!(gen_fn, args, assmt_or_selection, key, state)
    end
end
