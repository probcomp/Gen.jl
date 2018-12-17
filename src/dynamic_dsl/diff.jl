############
# calldiff #
############

get_call_diff(state, key) = get_leaf_node(state.calldiffs, key)

##############
# choicediff #
##############

function set_choice_diff_no_prev!(state, key)
    choicediff = NewChoiceDiff()
    set_leaf_node!(state.choicediffs, key, choicediff)
end

function set_choice_diff!(state, key, value_changed::Bool,
                          prev_retval::T) where {T}
    if value_changed
        choicediff = PrevChoiceDiff(prev_retval)
    else
        choicediff = NoChoiceDiff()
    end
    set_leaf_node!(state.choicediffs, key, choicediff)
end

get_choice_diff(state, key) = get_leaf_node(state.choicediffs, key)
