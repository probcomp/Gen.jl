###################
# address schemas #
###################

abstract type AddressSchema end

struct StaticAddressSchema <: AddressSchema
    # TODO can add type information
    leaf_nodes::Set{Symbol}
    internal_nodes::Set{Symbol}
end

leaf_node_keys(schema::StaticAddressSchema) = schema.leaf_nodes
internal_node_keys(schema::StaticAddressSchema) = schema.internal_nodes

# choice tries that have static address schemas should also support faster
# accessors, which make the address explicit in the type (Val(:foo) instaed of
# :foo)
function static_get_leaf_node end
function static_get_internal_node end
export static_get_leaf_node
export static_get_internal_node

struct VectorAddressSchema <: AddressSchema end 
struct SingleDynamicKeyAddressSchema <: AddressSchema end 
struct DynamicAddressSchema <: AddressSchema end 
struct EmptyAddressSchema <: AddressSchema end

export AddressSchema
export StaticAddressSchema
export VectorAddressSchema
export SingleDynamicKeyAddressSchema
export DynamicAddressSchema
export EmptyAddressSchema


#########################
# choice trie interface #
#########################

"""
    get_address_schema(T)

Return a shallow, compile-time address schema.
    
    Base.isempty(choices)

Are there any primitive random choices anywhere in the hierarchy?

    has_internal_node(choices, addr)

    get_internal_node(choices, addr)

    get_internal_nodes(choices)

Return an iterator over top-level internal nodes

    has_leaf_node(choices, addr)

    get_leaf_node(choices, addr)
    
    get_leaf_nodes(choices)

Return an iterator over top-level leaf nodes

    Base.haskey(choices, addr)

Alias for has_leaf_node

    Base.getindex(choices, addr)

Alias for get_leaf_node
"""
abstract type ChoiceTrie end

# defaults
has_internal_node(trie::ChoiceTrie, addr) = false
get_internal_node(trie::ChoiceTrie, addr) = throw(KeyError(addr))
has_leaf_node(trie::ChoiceTrie, addr) = false
get_leaf_node(trie::ChoiceTrie, addr) = throw(KeyError(addr))
Base.haskey(trie::ChoiceTrie, addr) = has_leaf_node(trie, addr)
Base.getindex(trie::ChoiceTrie, addr) = get_leaf_node(trie, addr)

function to_nested_dict(choice_trie::ChoiceTrie)
    dict = Dict()
    for (key, value) in get_leaf_nodes(choice_trie)
        dict[key] = value
    end
    for (key, node) in get_internal_nodes(choice_trie)
        dict[key] = to_nested_dict(node)
    end
    dict
end

import JSON
Base.print(trie::ChoiceTrie) = JSON.print(to_nested_dict(trie), 4)

export ChoiceTrie
export get_address_schema
export has_internal_node
export get_internal_node
export get_internal_nodes
export has_leaf_node
export get_leaf_node
export get_leaf_nodes 

function _fill_array! end
function _from_array end

function to_array(trie::ChoiceTrie, ::Type{T}) where {T}
    arr = Vector{T}(undef, 32)
    n = _fill_array!(trie, arr, 0)
    @assert n <= length(arr)
    resize!(arr, n)
    arr
end

function from_array(proto_trie::ChoiceTrie, arr)
    (n, trie) = _from_array(proto_trie, arr, 0)
    if n != length(arr)
        error("Dimension mismatch: $n, $(length(arr))")
    end
    trie
end

export to_array, from_array



######################
# static choice trie #
######################

struct StaticChoiceTrie{R,S,T,U} <: ChoiceTrie
    leaf_nodes::NamedTuple{R,S}
    internal_nodes::NamedTuple{T,U}
end

# TODO invariant: all internal_nodes are nonempty, but this is not verified at construction time

function get_address_schema(::Type{StaticChoiceTrie{R,S,T,U}}) where {R,S,T,U}
    leaf_nodes = Set{Symbol}()
    internal_nodes = Set{Symbol}()
    for (key, _) in zip(R, S.parameters)
        push!(leaf_nodes, key)
        #leaf_nodes[key] = typ
    end
    for (key, _) in zip(T, U.parameters)
        push!(internal_nodes, key)
        #internal_nodes[key] = typ
    end
    StaticAddressSchema(leaf_nodes, internal_nodes)
end

function Base.isempty(trie::StaticChoiceTrie)
    length(trie.leaf_nodes) == 0 && length(trie.internal_nodes) == 0
end

get_leaf_nodes(trie::StaticChoiceTrie) = pairs(trie.leaf_nodes)
get_internal_nodes(trie::StaticChoiceTrie) = pairs(trie.internal_nodes)

function _has_internal_node(trie::StaticChoiceTrie, ::Val{A}) where {A}
    haskey(trie.internal_nodes, A)
end

function has_internal_node(trie::StaticChoiceTrie, key::Symbol)
    haskey(trie.internal_nodes, key)
end

function has_internal_node(trie::StaticChoiceTrie, addr::Pair)
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        node = trie.internal_nodes[first]
        has_internal_node(node, rest)
    else
        false
    end
end

function static_get_internal_node(trie::StaticChoiceTrie, ::Val{A}) where {A}
    trie.internal_nodes[A]
end

function get_internal_node(trie::StaticChoiceTrie, key::Symbol)
    trie.internal_nodes[key]
end

function get_internal_node(trie::StaticChoiceTrie, addr::Pair)
    (first, rest) = addr
    node = trie.internal_nodes[first]
    get_internal_node(node, rest)
end

function _has_leaf_node(trie::StaticChoiceTrie, ::Val{A}) where {A}
    haskey(trie.leaf_nodes, A)
end

function has_leaf_node(trie::StaticChoiceTrie, key::Symbol)
    haskey(trie.leaf_nodes, key)
end

function has_leaf_node(trie::StaticChoiceTrie, addr::Pair)
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        node = trie.internal_nodes[first]
        has_leaf_node(node, rest)
    else
        false
    end
end

function static_get_leaf_node(trie::StaticChoiceTrie, ::Val{A}) where {A}
    trie.leaf_nodes[A]
end

function get_leaf_node(trie::StaticChoiceTrie, key::Symbol)
    trie.leaf_nodes[key]
end

function get_leaf_node(trie::StaticChoiceTrie, addr::Pair)
    (first, rest) = addr
    node = trie.internal_nodes[first]
    get_leaf_node(node, rest)
end

# convert from any other schema that has only Val{:foo} addresses
function StaticChoiceTrie(other::ChoiceTrie)
    leaf_keys_and_nodes = collect(get_leaf_nodes(other))
    internal_keys_and_nodes = collect(get_internal_nodes(other))
    if length(leaf_keys_and_nodes) > 0
        (leaf_keys, leaf_nodes) = collect(zip(leaf_keys_and_nodes...))
        #leaf_keys = map((k) -> typeof(k).parameters[1]::Symbol, leaf_keys)
    else
        (leaf_keys, leaf_nodes) = ((), ())
    end
    if length(internal_keys_and_nodes) > 0
        (internal_keys, internal_nodes) = collect(zip(internal_keys_and_nodes...))
        #internal_keys = map((k) -> typeof(k).parameters[1]::Symbol, internal_keys)
    else
        (internal_keys, internal_nodes) = ((), ())
    end
    StaticChoiceTrie(
        NamedTuple{leaf_keys}(leaf_nodes),
        NamedTuple{internal_keys}(internal_nodes))
end

function pair(a, b, key1::Symbol, key2::Symbol)
    StaticChoiceTrie(NamedTuple(), NamedTuple{(key1,key2)}((a, b)))
end

function unpair(trie, key1::Symbol, key2::Symbol)
    if length(get_leaf_nodes(trie)) != 0 || length(get_internal_nodes(trie)) > 2
        error("Not a pair")
    end
    a = has_internal_node(trie, key1) ? get_internal_node(trie, key1) : EmptyChoiceTrie()
    b = has_internal_node(trie, key2) ? get_internal_node(trie, key2) : EmptyChoiceTrie()
    (a, b)
end

# TODO make it a generated function?
function _fill_array!(trie::StaticChoiceTrie, arr::Vector{T}, start_idx::Int) where {T}
    if length(arr) < start_idx + length(trie.leaf_nodes)
        resize!(arr, 2 * length(arr))
    end
    for (i, value) in enumerate(trie.leaf_nodes)
        arr[start_idx + i] = value
    end
    idx = start_idx + length(trie.leaf_nodes)
    for (i, node) in enumerate(trie.internal_nodes)
        n_written = _fill_array!(node, arr, idx)
        idx += n_written
    end
    idx - start_idx
end

@generated function _from_array(proto_trie::StaticChoiceTrie{R,S,T,U}, arr::Vector{V}, start_idx::Int) where {R,S,T,U,V}
    leaf_node_keys = proto_trie.parameters[1]
    leaf_node_types = proto_trie.parameters[2].parameters
    internal_node_keys = proto_trie.parameters[3]
    internal_node_types = proto_trie.parameters[4].parameters

    # leaf nodes
    leaf_node_refs = Expr[]
    for (i, key) in enumerate(leaf_node_keys)
        expr = Expr(:ref, :arr, Expr(:call, :(+), :start_idx, QuoteNode(i)))
        push!(leaf_node_refs, expr)
    end

    # internal nodes
    internal_node_blocks = Expr[]
    internal_node_names = Symbol[]
    for (key, typ) in zip(internal_node_keys, internal_node_types)
        node = gensym()
        push!(internal_node_names, node)
        push!(internal_node_blocks, quote
            (n_read, $node::$typ) = _from_array(proto_trie.internal_nodes.$key, arr, idx)
            idx += n_read 
        end)
    end

    quote
        n_read = $(QuoteNode(length(leaf_node_keys)))
        leaf_nodes_field = NamedTuple{R,S}(($(leaf_node_refs...),))
        idx::Int = start_idx + n_read
        $(internal_node_blocks...)
        internal_nodes_field = NamedTuple{T,U}(($(internal_node_names...),))
        trie = StaticChoiceTrie{R,S,T,U}(leaf_nodes_field, internal_nodes_field)
        (idx - start_idx, trie)
    end
end

export StaticChoiceTrie
export pair, unpair

#######################
# dynamic choice trie #
#######################

struct DynamicChoiceTrie <: ChoiceTrie
    leaf_nodes::Dict{Any,Any}
    internal_nodes::Dict{Any,Any}
end

DynamicChoiceTrie() = DynamicChoiceTrie(Dict(), Dict())
get_address_schema(::Type{DynamicChoiceTrie}) = DynamicAddressSchema()

# invariant: all internal nodes are nonempty
Base.isempty(trie::DynamicChoiceTrie) = isempty(trie.leaf_nodes) && isempty(trie.internal_nodes)
get_leaf_nodes(trie::DynamicChoiceTrie) = trie.leaf_nodes
get_internal_nodes(trie::DynamicChoiceTrie) = trie.internal_nodes

function Base.values(trie::DynamicChoiceTrie)
    iterators::Vector = collect(map(values, trie.internal_nodes))
    push!(iterators, values(trie.leaf_nodes))
    Iterators.flatten(iterators)
end

function has_internal_node(trie::DynamicChoiceTrie, addr)
    haskey(trie.internal_nodes, addr)
end

function has_internal_node(trie::DynamicChoiceTrie, addr::Pair)
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        node = trie.internal_nodes[first]
        has_internal_node(node, rest)
    else
        false
    end
end

function get_internal_node(trie::DynamicChoiceTrie, addr)
    trie.internal_nodes[addr]
end

function get_internal_node(trie::DynamicChoiceTrie, addr::Pair)
    (first, rest) = addr
    node = trie.internal_nodes[first]
    get_internal_node(node, rest)
end

function has_leaf_node(trie::DynamicChoiceTrie, addr)
    haskey(trie.leaf_nodes, addr)
end

function has_leaf_node(trie::DynamicChoiceTrie, addr::Pair)
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        node = trie.internal_nodes[first]
        has_leaf_node(node, rest)
    else
        false
    end
end

function get_leaf_node(trie::DynamicChoiceTrie, addr)
    trie.leaf_nodes[addr]
end

function get_leaf_node(trie::DynamicChoiceTrie, addr::Pair)
    (first, rest) = addr
    node = trie.internal_nodes[first]
    get_leaf_node(node, rest)
end

# mutation (not part of the choice trie interface)

function set_leaf_node!(trie::DynamicChoiceTrie, addr, value)
    trie.leaf_nodes[addr] = value
end

function set_leaf_node!(trie::DynamicChoiceTrie, addr::Pair, value)
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        node = trie.internal_nodes[first]
    else
        node = DynamicChoiceTrie()
        trie.internal_nodes[first] = node
    end
    node = trie.internal_nodes[first]
    set_leaf_node!(node, rest, value)
end

function set_internal_node!(trie::DynamicChoiceTrie, addr, new_node)
    if !isempty(new_node)
        if haskey(trie.internal_nodes, addr)
            error("Node already exists at $addr")
        end
        trie.internal_nodes[addr] = new_node
    end
end

function set_internal_node!(trie::DynamicChoiceTrie, addr::Pair, new_node)
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        node = trie.internal_nodes[first]
    else
        node = DynamicChoiceTrie()
        trie.internal_nodes[first] = node
    end
    set_internal_node!(node, rest, new_node)
end

Base.setindex!(trie::DynamicChoiceTrie, value, addr) = set_leaf_node!(trie, addr, value)

# b should implement the choice-trie interface
function Base.merge!(a::DynamicChoiceTrie, b)
    for (key, value) in get_leaf_nodes(b)
        a.leaf_nodes[key] = value
    end
    for (key, b_node) in get_internal_nodes(b)
        @assert !isempty(b_node)
        if haskey(a.internal_nodes, key)
            # NOTE: error if a.internal_nodes[key] does not implement merge!
            @assert !isempty(a.internal_nodes[key])
            merge!(a.internal_nodes[key], b_node)
        else
            a.internal_nodes[key] = b_node
        end
    end
    a
end

function _fill_array!(trie::DynamicChoiceTrie, arr::Vector{T}, start_idx::Int) where {T}
    if length(arr) < start_idx + length(trie.leaf_nodes)
        resize!(arr, 2 * length(arr))
    end
    leaf_keys_sorted = sort(collect(keys(trie.leaf_nodes)))
    internal_node_keys_sorted = sort(collect(keys(trie.internal_nodes)))
    for (i, key) in enumerate(leaf_keys_sorted)
        arr[start_idx + i] = trie.leaf_nodes[key]
    end
    idx = start_idx + length(trie.leaf_nodes)
    for key in internal_node_keys_sorted
        n_written = _fill_array!(get_internal_node(trie, key), arr, idx)
        idx += n_written
    end
    idx - start_idx
end

function _from_array(proto_trie::DynamicChoiceTrie, arr::Vector{T}, start_idx::Int) where {T}
    @assert length(arr) >= start_idx
    trie = DynamicChoiceTrie()
    leaf_keys_sorted = sort(collect(keys(proto_trie.leaf_nodes)))
    internal_node_keys_sorted = sort(collect(keys(proto_trie.internal_nodes)))
    for (i, key) in enumerate(leaf_keys_sorted)
        trie.leaf_nodes[key] = arr[start_idx + i]
    end
    idx = start_idx + length(trie.leaf_nodes)
    for key in internal_node_keys_sorted
        (n_read, node) = _from_array(get_internal_node(proto_trie, key), arr, idx)
        idx += n_read
        trie.internal_nodes[key] = node
    end
    (idx - start_idx, trie)
end

export DynamicChoiceTrie


#######################################
## vector combinator for choice tries #
#######################################

# TODO implement LeafVectorChoiceTrie, which stores a vector of leaf nodes

struct InternalVectorChoiceTrie{T} <: ChoiceTrie
    internal_nodes::Vector{T}
    is_empty::Bool
end

# note some internal nodes may be empty

get_address_schema(::Type{InternalVectorChoiceTrie}) = VectorAddressSchema()

function vectorize_internal(nodes::Vector{T}) where {T}
    is_empty = all(map(isempty, nodes))
    InternalVectorChoiceTrie(nodes, is_empty)
end

Base.isempty(choices::InternalVectorChoiceTrie) = choices.is_empty

function has_internal_node(choices::InternalVectorChoiceTrie, addr::Int)
    n = length(choices.internal_nodes)
    addr >= 1 && addr <= n
end

function has_internal_node(choices::InternalVectorChoiceTrie, addr::Pair)
    (first, rest) = addr
    node = choices.internal_nodes[first]
    has_internal_node(node, rest)
end

function get_internal_node(choices::InternalVectorChoiceTrie, addr::Int)
    choices.internal_nodes[addr]
end

function get_internal_node(choices::InternalVectorChoiceTrie, addr::Pair)
    (first, rest) = addr
    node = choices.internal_nodes[first]
    get_internal_node(node, rest)
end

function has_leaf_node(choices::InternalVectorChoiceTrie, addr::Pair)
    (first, rest) = addr
    node = choices.internal_nodes[first]
    has_leaf_node(node, rest)
end

function get_leaf_node(choices::InternalVectorChoiceTrie, addr::Pair)
    (first, rest) = addr
    node = choices.internal_nodes[first]
    get_leaf_node(node, rest)
end

function get_internal_nodes(choices::InternalVectorChoiceTrie)
    # TODO need to check they are not empty
    ((i, choices.internal_nodes[i]) for i=1:length(choices.internal_nodes))
end

get_leaf_nodes(choices::InternalVectorChoiceTrie) = ()

function _fill_array!(trie::InternalVectorChoiceTrie, arr::Vector{T}, start_idx::Int) where {T}
    idx = start_idx
    for key=1:length(trie.internal_nodes)
        n = _fill_array!(trie.internal_nodes[key], arr, idx)
        idx += n
    end
    idx - start_idx
end

function _from_array(proto_trie::InternalVectorChoiceTrie{U}, arr::Vector{T}, start_idx::Int) where {T,U}
    @assert length(arr) >= start_idx
    nodes = Vector{U}(undef, length(proto_trie.internal_nodes))
    idx = start_idx
    for key=1:length(proto_trie.internal_nodes)
        (n_read, nodes[key]) = _from_array(proto_trie.internal_nodes[key], arr, idx)
        idx += n_read
    end
    trie = InternalVectorChoiceTrie(nodes, proto_trie.is_empty)
    (idx - start_idx, trie)
end

export InternalVectorChoiceTrie
export vectorize_internal


#####################
# empty choice trie #
#####################

struct EmptyChoiceTrie <: ChoiceTrie end

Base.isempty(::EmptyChoiceTrie) = true
get_address_schema(::Type{EmptyChoiceTrie}) = EmptyAddressSchema()
get_internal_nodes(::EmptyChoiceTrie) = ()
get_leaf_nodes(::EmptyChoiceTrie) = ()

_fill_array!(trie::EmptyChoiceTrie, arr::Vector, start_idx::Int) = 0
_from_array(proto_trie::EmptyChoiceTrie, arr::Vector, start_idx::Int) = (0, EmptyChoiceTrie())

export EmptyChoiceTrie
