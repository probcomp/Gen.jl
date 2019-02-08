function process_all_retained!(gen_fn::Unfold{T,U}, params::Tuple, argdiff::UnfoldCustomArgDiff,
                               choices_or_selection, prev_length::Int, new_length::Int,
                               retained_and_targeted::Set{Int}, state) where {T,U}
    if argdiff.params_changed
        # visit every retained kernel application
        for key=1:min(prev_length,new_length)
            # TODO allow user to pass more specific argdiff information
            process_retained!(gen_fn, params, choices_or_selection,
                key, unknownargdiff, state)
        end
    else
        # visit only certain retained kernel applications
        to_visit::Vector{Int} = sort(collect(retained_and_targeted))
        local is_state_diff::Bool
        key = 0
        if argdiff.init_changed
            is_state_diff = true
            key = 1
            while is_state_diff && key <= min(prev_length, new_length)
                is_state_diff = process_retained!(gen_fn, params, choices_or_selection,
                    key, unknownargdiff, state)
                key += 1
            end
        end
        for i=1:length(to_visit)
            if key > to_visit[i]
                # we have already visited it
                continue
            end
            is_state_diff = true
            key = to_visit[i]
            while is_state_diff && key <= min(prev_length, new_length)
                is_state_diff = process_retained!(gen_fn, params, choices_or_selection,
                    key, unknownargdiff, state)
                key += 1
            end
        end
    end
end

"""
Process all new applications.
"""
function process_all_new!(gen_fn::Unfold{T,U}, params::Tuple, choices_or_selection,
                          prev_len::Int, new_len::Int, state) where {T,U}
    for key=prev_len+1:new_len
        process_new!(gen_fn, params, choices_or_selection, key, state)
    end
end
