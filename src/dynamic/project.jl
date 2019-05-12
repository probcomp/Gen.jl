function project_recurse(trie::Trie{Any,ChoiceOrCallRecord},
                         selection::Selection)
    weight = 0.
    for (key, choice_or_call) in get_leaf_nodes(trie)
        if choice_or_call.is_choice
            if key in selection
                weight += choice_or_call.score
            end
        else
            subselection = selection[key]
            weight += project(choice_or_call.subtrace_or_retval, subselection)
        end
    end
    for (key, subtrie) in get_internal_nodes(trie)
        subselection = selection[key]
        weight += project_recurse(subtrie, subselection)
    end
    weight
end

function project(trace::DynamicDSLTrace, selection::Selection)
    project_recurse(trace.trie, selection)
end

project(trace::DynamicDSLTrace, ::EmptySelection) = trace.noise
