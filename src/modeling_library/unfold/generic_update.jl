function process_all_retained!(gen_fn::Unfold{T,U}, params::Tuple, argdiffs::Tuple,
                               choices_or_selection, prev_length::Int, new_length::Int,
                               retained_and_targeted::Set{Int}, state) where {T,U}

    len_diff = argdiffs[1]
    init_state_diff = argdiffs[2]
    param_diffs = argdiffs[3:end] # a tuple of diffs

    if any(diff != NoChange() for diff in param_diffs)

        # visit every retained kernel application
        state_diff = init_state_diff
        for key=1:min(prev_length,new_length)
            state_diff = process_retained!(gen_fn, params, choices_or_selection,
                key, (NoChange(), state_diff, param_diffs...), state)
        end

    else
        # every parameter diff is NoChange()

        # visit only certain retained kernel applications
        to_visit::Vector{Int} = sort(collect(retained_and_targeted))
        key = 0
        state_diff = init_state_diff
        if state_diff != NoChange()
            key = 1
            visit = true
            while visit && key <= min(prev_length, new_length)
                state_diff = process_retained!(gen_fn, params, choices_or_selection,
                    key, (NoChange(), state_diff, param_diffs...), state)
                key += 1
                visit = (state_diff != NoChange())
            end
        end
        for i=1:length(to_visit)
            if key > to_visit[i]
                # we have already visited it
                continue
            end
            key = to_visit[i]
            visit = true
            while visit && key <= min(prev_length, new_length)
                state_diff = process_retained!(gen_fn, params, choices_or_selection,
                    key, (NoChange(), state_diff, param_diffs...), state)
                key += 1
                visit = (state_diff != NoChange())
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
