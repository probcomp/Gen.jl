struct StaticAddressTree{LeafType, Addrs, SubtreeTypes} <: AddressTree{LeafType}
    subtrees::NamedTuple{Addrs, SubtreeTypes}
    function StaticAddressTree{LeafType}(nt::NamedTuple{Addrs, Subtrees}) where {
        LeafType, Addrs, Subtrees <: Tuple{Vararg{<:AddressTree{<:LeafType}}}
    }
        new{LeafType, Addrs, Subtrees}(nt)
    end
end

# If some of the subtrees are `EmptyAddressTree`s, but the `LeafType` does not have `EmptyAddressTree`
# as a subtype, strip away the addresses which point to EmptyAddressTrees.
# TODO: is this the implementation we want?  is the performance hit for removing the empty subtrees here worthwhile?
# also TODO: should I make this @generated?
function StaticAddressTree{LeafType}(nt::NamedTuple{Addrs, Subtrees}) where {
    LeafType, Addrs, Subtrees <: Tuple{Vararg{<:Union{AddressTree{<:LeafType}, EmptyAddressTree}}}
}
    #@assert Subtrees <: Tuple{Vararg{<:AddressTree{<:Union{LeafType, EmptyAddressTree}}}}
    # if @generated
    # println(Subtrees.parameters)
    #     nonempty_indices = (x -> x != EmptyAddressTree, Subtrees.parameters)
    #     nonempty_addrs = Tuple(Addrs[i] for i in nonempty_indices)
    #     nonempty_subtrees_exprs = [:(nt[$addr]) for addr in nonempty_addrs]
    #     nonempty_subtrees_expr = :((nonempty_subtrees_exprs...,))
    #     quote StaticAddressTree{$LeafType}(NamedTuple{$nonempty_addrs}($nonempty_subtrees_expr)) end
    # else
        nonempty_addrs = Tuple(findall(x -> x != EmptyAddressTree(), nt))
        nonempty_subtrees = Tuple(nt[addr] for addr in nonempty_addrs)
        StaticAddressTree{LeafType}(NamedTuple{nonempty_addrs}(nonempty_subtrees))
    # end
end

# NOTE: It is probably better to avoid using this constructor when possible since I suspect it is less performant
# than if we specify `LeafType`.
function StaticAddressTree(subtrees::NamedTuple{Addrs, SubtreeTypes}) where {Addrs, SubtreeTypes <: Tuple{Vararg{AddressTree}}}
    uniontype = Union{SubtreeTypes.parameters...}
    if @generated
        quote StaticAddressTree{$uniontype}(subtrees) end
    else
        StaticAddressTree{uniontype}(subtrees)
    end
end
"""
    StaticAddressTree{LeafType}(; a=val, b=tree, ...)
    StaticAddressTree(; a=val, b=tree, ...)

Construct a static address tree with the given address-subtree
or address-value pairs.  (The addresses must be top-level symbols;
if the RHS is an AddressTree, this will be the subtree; if not, the
subtree will be a `Value` with the given value.)
"""
StaticAddressTree(;addrs_to_vals_and_trees...) = StaticAddressTree(addrs_subtrees_namedtuple(addrs_to_vals_and_trees))
StaticAddressTree{LeafType}(; addrs_to_vals_and_trees...) where {LeafType} = StaticAddressTree{LeafType}(addrs_subtrees_namedtuple(addrs_to_vals_and_trees))

function addrs_subtrees_namedtuple(addrs_to_vals_and_trees)
    addrs = Tuple(addr for (addr, val_or_map) in addrs_to_vals_and_trees)
    trees = Tuple(val_or_map isa AddressTree ? val_or_map : Value(val_or_map) for (addr, val_or_map) in addrs_to_vals_and_trees)
    NamedTuple{addrs}(trees)
end

@inline get_subtrees_shallow(t::StaticAddressTree) = pairs(t.subtrees)
@inline get_submap(t::StaticAddressTree, addr::Pair) = _get_subtree(t, addr)

function get_subtree(t::StaticAddressTree{LeafType, Addrs}, addr::Symbol) where {LeafType, Addrs}
    if addr in Addrs
        t.subtrees[addr]
    else
        EmptyAddressTree()
    end
end

@generated function static_get_subtree(t::StaticAddressTree{LeafType, Addrs}, ::Val{A}) where {A, Addrs, LeafType}
    if A in Addrs
        quote t.subtrees[A] end
    else
        quote EmptyAddressTree() end
    end
end
@inline static_get_subtree(::EmptyAddressTree, ::Val) = EmptyAddressTree()
@inline static_get_subtree(::Value, ::Val) = EmptyAddressTree()
@inline static_get_subtree(::AllSelection, ::Val) = AllSelection()

@inline static_get_value(choices::StaticAddressTree, v::Val) = get_value(static_get_subtree(choices, v))
@inline static_get_value(::EmptyAddressTree, ::Val) = throw(ChoiceMapGetValueError())

# convert a nonvalue choicemap all of whose top-level-addresses 
# are symbols into a staticchoicemap at the top level
StaticAddressTree(t::StaticAddressTree) = t
function StaticAddressTree(other::AddressTree{LeafType}) where {LeafType}
    keys_and_nodes = get_subtrees_shallow(other)
    if length(keys_and_nodes) > 0
        addrs = Tuple(key for (key, _) in keys_and_nodes)
        submaps = Tuple(submap for (_, submap) in keys_and_nodes)
    else
        addrs = ()
        submaps = ()
    end
    StaticAddressTree{LeafType}(NamedTuple{addrs}(submaps))
end
StaticAddressTree(::AllSelection) = AllSelection()
StaticAddressTree(::EmptyAddressTree) = EmptyAddressTree()
StaticAddressTree(v::Value) = v
StaticAddressTree(::AddressTreeLeaf) = error("Not implemented")
StaticAddressTree{LeafType}(::NamedTuple{(),Tuple{}}) where {LeafType} = EmptyAddressTree()
StaticAddressTree(::NamedTuple{(),Tuple{}}) = EmptyAddressTree()
StaticAddressTree{LeafType}(other::AddressTree{<:LeafType}) where {LeafType} = StaticAddressTree(other)

# TODO: deep conversion to static choicemap

"""
    tree = pair(tree1::AddressTree, tree2::AddressTree, key1::Symbol, key2::Symbol)

Return an address tree that contains `tree1` as a subtree under `key1`
and `tree2` as a subtree under `key2`.
"""
function pair(tree1::AddressTree, tree2::AddressTree, key1::Symbol, key2::Symbol)
    StaticAddressTree(NamedTuple{(key1, key2)}((tree1, tree2)))
end

"""
    (tree1, tree2) = unpair(tree::AddressTree, key1::Symbol, key2::Symbol)

Return the two subtrees at `key1` and `key2`, one or both of which may be empty.

It is an error if there are any subtrees at keys other than `key1` and `key2`.
"""
function unpair(tree::AddressTree, key1::Symbol, key2::Symbol)
    if length(collect(get_subtrees_shallow(tree))) != 2
        error("Not a pair")
    end
    (get_subtree(tree, key1), get_subtree(tree, key2))
end

@generated function Base.merge(tree1::StaticAddressTree{T1, Addrs1, SubmapTypes1},
    tree2::StaticAddressTree{T2, Addrs2, SubmapTypes2}) where {T1, T2, Addrs1, Addrs2, SubmapTypes1, SubmapTypes2}
    
    addr_to_type1 = Dict{Symbol, Type{<:AddressTree}}()
    addr_to_type2 = Dict{Symbol, Type{<:AddressTree}}()
    for (i, addr) in enumerate(Addrs1)
        addr_to_type1[addr] = SubmapTypes1.parameters[i]
    end
    for (i, addr) in enumerate(Addrs2)
        addr_to_type2[addr] = SubmapTypes2.parameters[i]
    end

    merged_addrs = Tuple(union(Set(Addrs1), Set(Addrs2)))
    submap_exprs = []

    for addr in merged_addrs
        type1 = get(addr_to_type1, addr, EmptyAddressTree)
        type2 = get(addr_to_type2, addr, EmptyAddressTree)

        if type1 <: EmptyAddressTree
            push!(submap_exprs, 
                quote tree2.subtrees.$addr end
            )
        elseif type2 <: EmptyAddressTree
            push!(submap_exprs,
                quote tree1.subtrees.$addr end
            )
        else
            push!(submap_exprs,
                quote merge(tree1.subtrees.$addr, tree2.subtrees.$addr) end
            )
        end
    end

    leaftype = Union{T1, T2}

    quote
        StaticAddressTree{$leaftype}(NamedTuple{$merged_addrs}(($(submap_exprs...),)))
    end
end

@generated function _from_array(proto_choices::StaticAddressTree{LT, Addrs, SubmapTypes},
    arr::Vector{T}, start_idx::Int) where {LT, T, Addrs, SubmapTypes}

    perm = sortperm(collect(Addrs))
    sorted_addrs = Addrs[perm]
    submap_var_names = Vector{Symbol}(undef, length(sorted_addrs))

    exprs = [quote idx = start_idx end]

    for (idx, addr) in zip(perm, sorted_addrs)
        submap_var_name = gensym(addr)
        submap_var_names[idx] = submap_var_name
        push!(exprs,
            quote
                (n_read, $submap_var_name) = _from_array(proto_choices.subtrees.$addr, arr, idx)
                idx += n_read
            end
        )
    end

    quote
        $(exprs...)
        submaps = NamedTuple{Addrs}(( $(submap_var_names...), ))
        choices = StaticAddressTree{LT}(submaps)
        (idx - start_idx, choices)
    end
end

function get_address_schema(::Type{StaticAddressTree{LT, Addrs, SubtreeTypes}}) where {LT, Addrs, SubtreeTypes}
    StaticAddressSchema(Set(Addrs))
end

export StaticAddressTree
export pair, unpair
export static_get_subtree, static_get_value