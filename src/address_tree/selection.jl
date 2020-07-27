const Selection = AddressTree{<:Union{SelectionLeaf, EmptyAddressTree}}

const StaticSelection = StaticAddressTree{SelectionLeaf}
const EmptySelection = EmptyAddressTree

"""
    in(addr, selection::Selection)

Whether the address is selected in the given selection.
"""
@inline function Base.in(addr, selection::Selection) 
    get_subtree(selection, addr) === AllSelection()
end

# indexing returns subtrees for selections
Base.getindex(selection::AddressTree{SelectionLeaf}, addr) = get_subtree(selection, addr)

# TODO: deprecate indexing syntax and only use this
get_subselection(s::Selection, addr) = get_subtree(s, addr)

get_subselections(s::Selection) = get_subtrees_shallow(s)

Base.merge(::AllSelection, ::Selection) = AllSelection()
Base.merge(::Selection, ::AllSelection) = AllSelection()
Base.merge(::AllSelection, ::AllSelection) = AllSelection()
Base.merge(::AllSelection, ::EmptySelection) = AllSelection()
Base.merge(::EmptySelection, ::AllSelection) = AllSelection()

"""
    filtered = SelectionFilteredAddressTree(tree, selection)

An address tree containing only the nodes in `tree` whose addresses are selected
in `selection.`
"""
struct SelectionFilteredAddressTree{T} <: AddressTree{T}
    tree::AddressTree{T}
    sel::Selection
end
SelectionFilteredAddressTree(t::AddressTree, ::AllSelection) = t
SelectionFilteredAddressTree(t::AddressTreeLeaf, ::AllSelection) = t
SelectionFilteredAddressTree(::AddressTree, ::EmptyAddressTree) = EmptyAddressTree()
SelectionFilteredAddressTree(::AddressTreeLeaf, ::EmptyAddressTree) = EmptyAddressTree()
SelectionFilteredAddressTree(::AddressTreeLeaf, ::Selection) = EmptyAddressTree() # if we hit a leaf node before a selected value, the node is not selected

function get_subtree(t::SelectionFilteredAddressTree, addr)
    subselection = get_subtree(t.sel, addr)
    if subselection === EmptyAddressTree()
        EmptyAddressTree()
    else
        SelectionFilteredAddressTree(get_subtree(t.tree, addr), subselection)
    end
end

function get_subtrees_shallow(t::SelectionFilteredAddressTree)
    nonempty_subtree_itr(
        (addr, SelectionFilteredAddressTree(subtree, get_subtree(t.sel, addr)))
        for (addr, subtree) in get_subtrees_shallow(t.tree)
    )
end

"""
    selected = get_selected(tree::AddressTree, selection::Selection)

Filter the address tree `tree` to only include leaf nodes at selected
addresses.
"""
get_selected(tree::AddressTree, selection::Selection) = SelectionFilteredAddressTree(tree, selection)

"""
    struct DynamicSelection <: HierarchicalSelection .. end
A hierarchical, mutable, selection with arbitrary addresses.
Can be mutated with the following methods:
    Base.push!(selection::DynamicSelection, addr)
Add the address and all of its sub-addresses to the selection.
Example:
```julia
selection = select()
@assert !(:x in selection)
push!(selection, :x)
@assert :x in selection
```
    set_subselection!(selection::DynamicSelection, addr, other::Selection)
Change the selection status of the given address and its sub-addresses that defined by `other`.
Example:
```julia
selection = select(:x)
@assert :x in selection
subselection = select(:y)
set_subselection!(selection, :x, subselection)
@assert (:x => :y) in selection
@assert !(:x in selection)
```
Note that `set_subselection!` does not copy data in `other`, so `other` may be mutated by a later calls to `set_subselection!` for addresses under `addr`.
"""
const DynamicSelection = DynamicAddressTree{SelectionLeaf}
set_subselection!(s::DynamicSelection, addr, sub::Selection) = set_subtree!(s, addr, sub)

function Base.push!(s::DynamicSelection, addr)
    set_subtree!(s, addr, AllSelection())
end
function Base.push!(s::DynamicSelection, addr::Pair)
    first, rest = addr
    subtree = get_subtree(s, first)
    if subtree isa DynamicSelection
        push!(subtree, rest)
    else
        new_subtree = select(rest)
        merge!(new_subtree, subtree)
        set_subtree!(s, first, new_subtree)
    end
end

function select(addrs...)
    selection = DynamicSelection()
    for addr in addrs
        set_subtree!(selection, addr, AllSelection())
    end
    selection
end

"""
    AddressSelection(::AddressTree)

A selection containing all of the addresses in the given address tree with a nonempty leaf node.
"""
struct AddressSelection{T} <: AddressTree{AllSelection}
    a::T
    AddressSelection(a::T) where {T <: AddressTree} = new{T}(a)
end
AddressSelection(::AddressTreeLeaf) = AllSelection()
AddressSelection(::EmptyAddressTree) = EmptyAddressTree()
get_subtree(a::AddressSelection, addr) = AddressSelection(get_subtree(a.a, addr))
function get_subtrees_shallow(a::AddressSelection)
    nonempty_subtree_itr((addr, AddressSelection(subtree)) for (addr, subtree) in get_subtrees_shallow(a.a))
end
get_address_schema(::Type{AddressSelection{T}}) where {T} = get_address_schema(T)

"""
    addrs(::AddressTree)

Returns a selection containing all of the addresses in the tree with a nonempty leaf node.
"""
addrs(a::AddressTree) = AddressSelection(a)

"""
    invert(sel::Selection)
    InvertedSelection(sel::Selection)

"Inverts" `sel` by transforming every `AllSelection` subtree
to an `EmptySelection` and every `EmptySelection` to an `AllSelection`.
"""
invert(sel::Selection) = InvertedSelection(sel)
struct InvertedSelection <: SelectionLeaf
    sel::Selection
end
InvertedSelection(::AllSelection) = EmptySelection()
InvertedSelection(::EmptySelection) = AllSelection()
get_subtree(s::InvertedSelection, address) = InvertedSelection(get_subtree(s.sel, address))
# get_subtrees_shallow uses default implementation for ::AddressTreeLeaf to return ()

export select, get_selected, addrs, get_subselection, get_subselections, invert
export Selection, DynamicSelection, EmptySelection, StaticSelection, InvertedSelection