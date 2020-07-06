function project_recurse(trie::Trie{Any, CallRecord},
                         selection::Selection)
    weight = 0.
    for (key, call) in get_leaf_nodes(trie)
        subselection = get_subselection(selection, key)
        weight += project(call.subtrace, subselection)
    end
    for (key, subtrie) in get_internal_nodes(trie)
        subselection = get_subselection(selection, key)
        weight += project_recurse(subtrie, subselection)
    end
    weight
end

function project(trace::DynamicDSLTrace, selection::Selection)
    project_recurse(trace.trie, selection)
end

project(trace::DynamicDSLTrace, ::EmptySelection) = trace.noise
