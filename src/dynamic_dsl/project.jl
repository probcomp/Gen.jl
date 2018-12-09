function project_recurse(choices::Trie{Any,ChoiceRecord},
                         selection::AddressSet)
    weight = 0.
    for (key, choice) in get_leaf_nodes(choices)
        if has_leaf_node(selection, key)
            weight += choice.score
        end
    end
    for (key, subchoices) in get_internal_nodes(choices)
        if has_internal_node(selection, key)
            subselection = get_internal_node(selection, key)
            @assert !isempty(subselection)
            weight += project_recurse(subchoices, subselection)
        end
    end
    weight
end

function project_recurse(calls::Trie{Any,CallRecord},
                         selection::AddressSet)
    weight = 0.
    for (key, subselection) in get_internal_nodes(selection)
        if has_internal_node(calls, key)
            subcalls = get_internal_node(calls, key)
            weight += project_recurse(subcalls, subselection)
        end
    end
    weight
end

function project(trace::DynamicDSLTrace, selection::AddressSet)
    (project_recurse(trace.choices, selection) +
     project_recurse(trace.calls, selection))
end

project(trace::DynamicDSLTrace, ::EmptyAddressSet) = trace.noise
