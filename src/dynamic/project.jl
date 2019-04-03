function project_recurse(trie::Trie{Any,ChoiceOrCallRecord},
                         selection::AddressSet)
    weight = 0.
    for (key, choice_or_call) in get_leaf_nodes(trie)
        if choice_or_call.is_choice
            if has_leaf_node(selection, key)
                weight += choice_or_call.score
            elseif has_internal_node(selection, key)
                error("Got internal selection node for choice at key $key")
            end
        else
            if has_internal_node(selection, key)
                subselection = get_internal_node(selection, key)
            elseif has_leaf_node(selection, key)
                error("Got leaf selection node for choice map at $key") # TODO handle this
            else
                subselection = EmptyAddressSet()
            end
            weight += project(call.subtrace, subselection)
        end
    end
    for (key, subtrie) in get_internal_nodes(trie)
        if has_internal_node(selection, key)
            subselection = get_internal_node(selection, key)
            @assert !isempty(subselection) # otherwise it would not has_internal_node
            weight += project_recurse(subtrie, subselection)
        end
    end
    weight
end

function project(trace::DynamicDSLTrace, selection::AddressSet)
    project_recurse(trace.trie, selection)
end

project(trace::DynamicDSLTrace, ::EmptyAddressSet) = trace.noise
