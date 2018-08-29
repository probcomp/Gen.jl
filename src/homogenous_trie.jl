##################
# HomogenousTrie #
##################

struct HomogenousTrie{K,V} <: ChoiceTrie
    leaf_nodes::Dict{K,V}
    internal_nodes::Dict{K,HomogenousTrie{K,V}}
end

HomogenousTrie{K,V}() where {K,V} = HomogenousTrie(Dict{K,V}(), Dict{K,HomogenousTrie{K,V}}())

# copy constructor for something supported read-only trie interface
function HomogenousTrie(other)
    trie = HomogenousTrie{Any,Any}()
    for (key, value) in get_leaf_nodes(other)
        set_leaf_node!(trie, key, value)
    end
    for (key, node) in get_internal_nodes(other)
        sub_trie = HomogenousTrie(node)
        set_internal_node!(trie, key, sub_trie)
    end
    trie
end

# TODO come up with a better printing method (and something nice for Jupyter
# notebooks)
import JSON
Base.println(trie::HomogenousTrie) = JSON.print(trie, 4)

# invariant: all internal nodes are nonempty
Base.isempty(trie::HomogenousTrie) = isempty(trie.leaf_nodes) && isempty(trie.internal_nodes)
get_leaf_nodes(trie::HomogenousTrie) = trie.leaf_nodes
get_internal_nodes(trie::HomogenousTrie) = trie.internal_nodes

function Base.values(trie::HomogenousTrie)
    iterators = convert(Vector{Any}, collect(map(values, values(trie.internal_nodes))))
    push!(iterators, values(trie.leaf_nodes))
    Iterators.flatten(iterators)
end

function has_internal_node(trie::HomogenousTrie, addr)
    haskey(trie.internal_nodes, addr)
end

function get_internal_node(trie::HomogenousTrie, addr)
    trie.internal_nodes[addr]
end

function set_internal_node!(trie::HomogenousTrie{K,V}, addr, new_node::HomogenousTrie{K,V}) where {K,V}
    if !isempty(new_node)
        trie.internal_nodes[addr] = new_node
    end
end

function set_internal_node!(trie::HomogenousTrie{K,V}, addr::Pair, new_node::HomogenousTrie{K,V}) where {K,V}
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        node = trie.internal_nodes[first]
    else
        node = HomogenousTrie{K,V}()
        trie.internal_nodes[first] = node
    end
    set_internal_node!(node, rest, new_node)
end

function delete_internal_node!(trie::HomogenousTrie, addr)
    delete!(trie.internal_nodes, addr)
    return isempty(trie.leaf_nodes) && isempty(trie.internal_nodes)
end

function delete_internal_node!(trie::HomogenousTrie, addr::Pair)
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        node = trie.internal_nodes[first]
        if delete_internal_node!(node, rest, new_node)
            delete!(trie.internal_nodes, first)
        end
    end
    return isempty(trie.leaf_nodes) && isempty(trie.internal_nodes)
end

function has_leaf_node(trie::HomogenousTrie, addr)
    haskey(trie.leaf_nodes, addr)
end

function get_leaf_node(trie::HomogenousTrie, addr)
    trie.leaf_nodes[addr]
end

function set_leaf_node!(trie::HomogenousTrie, addr, value)
    trie.leaf_nodes[addr] = value
end

function set_leaf_node!(trie::HomogenousTrie{K,V}, addr::Pair, value) where {K,V}
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        node = trie.internal_nodes[first]
    else
        node = HomogenousTrie{K,V}()
        trie.internal_nodes[first] = node
    end
    node = trie.internal_nodes[first]
    set_leaf_node!(node, rest, value)
end

function delete_leaf_node!(trie::HomogenousTrie, addr)
    delete!(trie.leaf_nodes, addr)
    return isempty(trie.leaf_nodes) && isempty(trie.internal_nodes)
end

function delete_leaf_node!(trie::HomogenousTrie, addr::Pair)
    (first, rest) = addr
    if haskey(trie.internal_nodes[first])
        node = trie.internal_nodes[first]
        if delete_leaf_node!(node, rest)
            delete!(trie.internal_nodes, first)
        end
    end
    return isempty(trie.leaf_nodes) && isempty(trie.internal_nodes)
end

Base.setindex!(trie::HomogenousTrie, value, addr) = set_leaf_node!(trie, addr, value)

function Base.merge!(a::HomogenousTrie{K,V}, b::HomogenousTrie{K,V}) where {K,V}
    merge!(a.leaf_nodes, b.leaf_nodes)
    for (key, a_sub) in a.sub
        if haskey(b.sub, key)
            b_sub = b.sub[key]
            merge!(a_sub, b_sub)
        end
    end
    for (key, b_sub) in b.sub
        if !haskey(a.sub, key)
            a.sub[key] = b_sub
        end
    end
    a
end

function Base.delete!(trie::HomogenousTrie, addrs::AddressSet)
    for key in get_leaf_nodes(addrs)
        delete_leaf_node!(trie, key)
        delete_internal_node!(trie, key)
    end
    for (key, addrs_node) in get_internal_nodes(addrs)
        if has_internal_node(trie, key)
            trie_node = get_internal_node(trie, key)
            if delete!(trie_node, addrs_node)
                delete!(trie.internal_nodes, key)
            end
        end
    end
    return isempty(trie)
end

get_address_schema(::HomogenousTrie) = DynamicSchema()

export HomogenousTrie
export set_internal_node!
export delete_internal_node!
export set_leaf_node!
export delete_leaf_node!
