"""
    struct DynamicAddressTree <: AddressTree .. end

A mutable AddressTree.

    tree = DynamicAddressTree()

Construct an empty address tree.

"""
struct DynamicAddressTree{LeafType} <: AddressTree{LeafType}
    subtrees::Dict{Any, AddressTree{<:LeafType}}
    function DynamicAddressTree{LeafType}() where {LeafType}
        new{LeafType}(Dict{Any, AddressTree}())
    end
end

"""
    tree = address_tree()

Construct an empty, mutable address tree.
"""
address_tree() = DynamicAddressTree{Any}()

get_address_schema(::Type{<:DynamicAddressTree}) = DynamicAddressTree

@inline get_subtrees_shallow(t::DynamicAddressTree) = t.subtrees
@inline get_subtree(t::DynamicAddressTree, addr) = get(t.subtrees, addr, EmptyAddressTree())
@inline get_subtree(t::DynamicAddressTree, addr::Pair) = _get_subtree(t, addr)
@inline Base.isempty(t::DynamicAddressTree) = isempty(t.subtrees)

function set_subtree!(t::DynamicAddressTree, addr, new_node::AddressTree)
    delete!(t.subtrees, addr)
    if !isempty(new_node)
        t.subtrees[addr] = new_node
    end
end
function set_subtree!(t::DynamicAddressTree{T}, addr::Pair, new_node::AddressTree) where {T}
    (first, rest) = addr
    if !haskey(t.subtrees, first)
        t.subtrees[first] = DynamicAddressTree{T}()
    end
    set_subtree!(t.subtrees[first], rest, new_node)
end

"""
    tree = shallow_dynamic_copy(other::AddressTree)

Make a shallow `DynamicAddressTree` copy of the given address tree.
"""
function shallow_dynamic_copy(other::AddressTree{LeafType}) where {LeafType}
    tree = DynamicAddressTree{LeafType}()
    for (addr, subtree) in get_subtrees_shallow(other)
        set_subtree!(tree, addr, subtree)
    end
    tree
end

"""
    tree = deep_dynamic_copy(other::AddressTree)

Make a deep copy of the given address tree, where every non-leaf-node
is a `DynamicAddressTree`.
"""
function deep_dynamic_copy(other::AddressTree{LeafType}) where {LeafType}
    tree = DynamicAddressTree{LeafType}()
    for (addr, subtree) in get_subtrees_shallow(other)
        if subtree isa AddressTreeLeaf
            set_subtree!(tree, addr, subtree)
        else
            set_subtree!(tree, addr, DynamicAddressTree(subtree))
        end
    end
    tree
end

"""
    tree = DynamicAddressTree(other::AddressTree)

Shallowly convert an address tree to dynamic.
"""
DynamicAddressTree(t::AddressTree) = shallow_dynamic_copy(t)
DynamicAddressTree(t::DynamicAddressTree) = t
DynamicAddressTree{LeafType}(t::AddressTree{<:LeafType}) where {LeafType} = DynamicAddressTree(t)

function _from_array(proto_choices::DynamicAddressTree{LT}, arr::Vector{T}, start_idx::Int) where {T, LT}
    choices = DynamicAddressTree{LT}()
    keys_sorted = sort(collect(keys(proto_choices.subtrees)))
    idx = start_idx
    for key in keys_sorted
        (n_read, submap) = _from_array(proto_choices.subtrees[key], arr, idx)
        idx += n_read
        choices.subtrees[key] = submap
    end
    (idx - start_idx, choices)
end

export DynamicAddressTree, set_subtree!