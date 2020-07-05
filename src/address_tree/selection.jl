const Selection = AddressTree{AllSelection}

const StaticSelection = StaticAddressTree{AllSelection}
const DynamicSelection = DynamicAddressTree{AllSelection}
const EmptySelection = EmptyAddressTree

"""
    in(addr, selection::Selection)

Whether the address is selected in the given selection.
"""
@inline function Base.in(addr, selection::Selection) 
    get_subtree(selection, addr) === AllSelection()
end

# indexing returns subtrees for selections
function Base.getindex(selection::Selection, addr)
    get_subtree(selection, addr)
end

get_subselections(s::Selection) = get_subtrees_shallow(s)

function select(addrs...)
    selection = DynamicSelection()
    for addr in addrs
        set_subtree!(selection, addr, AllSelection())
    end
    selection
end

Base.merge(::AllSelection, ::Selection) = AllSelection()
Base.merge(::Selection, ::AllSelection) = AllSelection()
Base.merge(::AllSelection, ::AllSelection) = AllSelection()

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
    all_selected_including_empty = (
        (addr, SelectionFilteredAddressTree(subtree, get_subtree(t.sel, addr)))
        for (addr, subtree) in get_subtrees_shallow(t.tree)
    )

    return (
        (addr, tree) for (addr, tree) in all_selected_including_empty
        if tree !== EmptyAddressTree()
    )
end

"""
    selected = get_selected(tree::AddressTree, selection::Selection)

Filter the address tree `tree` to only include leaf nodes at selected
addresses.
"""
get_selected(tree::AddressTree, selection::Selection) = SelectionFilteredAddressTree(tree, selection)

export select, get_selected
export DynamicSelection, EmptySelection, StaticSelection