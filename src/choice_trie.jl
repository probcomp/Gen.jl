#########################
# choice trie interface #
#########################

"""
    get_address_schema(T)

Return a shallow, compile-time address schema.
    
    Base.isempty(trie)

Are there any primitive random choice anywhere in the hierarchy?

    has_internal_node(trie, addr)

    get_internal_node(trie, addr)

    get_internal_nodes(trie)

Return an iterator over top-level internal nodes

    has_leaf_node(trie, addr)

    get_leaf_node(trie, addr)
    
    get_leaf_nodes(trie)

Return an iterator over top-level leaf nodes

    Base.haskey(trie, addr)

Alias for has_leaf_node

    Base.getindex(trie, addr)

Alias for get_leaf_node
"""
abstract type ChoiceTrie end

has_internal_node(trie::ChoiceTrie, addr) = false
get_internal_node(trie::ChoiceTrie, addr) = throw(KeyError(addr))
has_leaf_node(trie::ChoiceTrie, addr) = false
get_leaf_node(trie::ChoiceTrie, addr) = throw(KeyError(addr))
Base.haskey(trie::ChoiceTrie, addr) = has_leaf_node(trie, addr)
Base.getindex(trie::ChoiceTrie, addr) = get_leaf_node(trie, addr)

# this code will be duplcated in:
# dynamic choice trie
# static choice trie
# basic block trace choices

function _has_leaf_node(choices::T, addr::Pair) where {T <: ChoiceTrie}
    (first, rest) = addr
    if has_internal_node(choices, first)
        node = get_internal_node(choices, first)
        has_leaf_node(node, rest)
    else
        false
    end
end

function _get_leaf_node(choices::T, addr::Pair) where {T <: ChoiceTrie}
    (first, rest) = addr
    node = get_internal_node(choices, first)
    get_leaf_node(node, rest)
end

function _has_internal_node(trie::T, addr::Pair) where {T <: ChoiceTrie}
    (first, rest) = addr
    if has_internal_node(trie, first)
        node = get_internal_node(trie, first)
        has_internal_node(node, rest)
    else
        false
    end
end

function _get_internal_node(trie::T, addr::Pair) where {T <: ChoiceTrie}
    (first, rest) = addr
    node = get_internal_node(trie, first)
    get_internal_node(node, rest)
end

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

function _print(io::IO, trie::ChoiceTrie, pre, vert_bars::Tuple)
    VERT = '\u2502'
    PLUS = '\u251C'
    HORZ = '\u2500'
    LAST = '\u2514'
    indent_vert = vcat(Char[' ' for _ in 1:pre], Char[VERT, '\n'])
    indent_vert_last = vcat(Char[' ' for _ in 1:pre], Char[VERT, '\n'])
    indent = vcat(Char[' ' for _ in 1:pre], Char[PLUS, HORZ, HORZ, ' '])
    indent_last = vcat(Char[' ' for _ in 1:pre], Char[LAST, HORZ, HORZ, ' '])
    for i in vert_bars
        indent_vert[i] = VERT
        indent[i] = VERT
        indent_last[i] = VERT
    end
    indent_vert_str = join(indent_vert)
    indent_vert_last_str = join(indent_vert_last)
    indent_str = join(indent)
    indent_last_str = join(indent_last)
    leaf_nodes = collect(get_leaf_nodes(trie))
    internal_nodes = collect(get_internal_nodes(trie))
    n = length(leaf_nodes) + length(internal_nodes)
    cur = 1
    for (key, value) in leaf_nodes
        print(io, indent_vert_str)
        print(io, (cur == n ? indent_last_str : indent_str) * "$(repr(key)) : $value\n")
        cur += 1
    end
    for (key, node) in internal_nodes
        print(io, indent_vert_str)
        print(io, (cur == n ? indent_last_str : indent_str) * "$(repr(key))\n")
        _print(io, node, pre + 4, cur == n ? (vert_bars...,) : (vert_bars..., pre+1))
        cur += 1
    end
end

function Base.print(io::IO, trie::ChoiceTrie)
    _print(io, trie, 0, ())
end

export ChoiceTrie
export get_address_schema
export has_internal_node
export get_internal_node
export get_internal_nodes
export has_leaf_node
export get_leaf_node
export get_leaf_nodes 

# choice tries that have static address schemas should also support faster
# accessors, which make the address explicit in the type (Val(:foo) instaed of
# :foo)
function static_get_leaf_node end
function static_get_internal_node end
export static_get_leaf_node
export static_get_internal_node

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
    leaf_keys = Set{Symbol}()
    internal_keys = Set{Symbol}()
    for (key, _) in zip(R, S.parameters)
        push!(leaf_keys, key)
    end
    for (key, _) in zip(T, U.parameters)
        push!(internal_keys, key)
    end
    StaticAddressSchema(leaf_keys, internal_keys)
end

function Base.isempty(trie::StaticChoiceTrie)
    length(trie.leaf_nodes) == 0 && length(trie.internal_nodes) == 0
end

get_leaf_nodes(trie::StaticChoiceTrie) = pairs(trie.leaf_nodes)
get_internal_nodes(trie::StaticChoiceTrie) = pairs(trie.internal_nodes)
has_leaf_node(trie::StaticChoiceTrie, addr::Pair) = _has_leaf_node(trie, addr)
get_leaf_node(trie::StaticChoiceTrie, addr::Pair) = _get_leaf_node(trie, addr)
has_internal_node(trie::StaticChoiceTrie, addr::Pair) = _has_internal_node(choices, addr)
get_internal_node(trie::StaticChoiceTrie, addr::Pair) = _get_internal_node(choices, addr)

# NOTE: there is no static_has_internal_node or static_has_leaf_node because
# this is known from the static address schema

## has_internal_node ##

function has_internal_node(trie::StaticChoiceTrie, key::Symbol)
    haskey(trie.internal_nodes, key)
end

## has_leaf_node ##

function has_leaf_node(trie::StaticChoiceTrie, key::Symbol)
    haskey(trie.leaf_nodes, key)
end

## get_internal_node ##

function get_internal_node(trie::StaticChoiceTrie, key::Symbol)
    trie.internal_nodes[key]
end

function static_get_internal_node(trie::StaticChoiceTrie, ::Val{A}) where {A}
    trie.internal_nodes[A]
end

## get_leaf_node ##

function get_leaf_node(trie::StaticChoiceTrie, key::Symbol)
    trie.leaf_nodes[key]
end

function static_get_leaf_node(trie::StaticChoiceTrie, ::Val{A}) where {A}
    trie.leaf_nodes[A]
end

# convert from any other schema that has only Val{:foo} addresses
function StaticChoiceTrie(other::ChoiceTrie)
    leaf_keys_and_nodes = collect(get_leaf_nodes(other))
    internal_keys_and_nodes = collect(get_internal_nodes(other))
    if length(leaf_keys_and_nodes) > 0
        (leaf_keys, leaf_nodes) = collect(zip(leaf_keys_and_nodes...))
    else
        (leaf_keys, leaf_nodes) = ((), ())
    end
    if length(internal_keys_and_nodes) > 0
        (internal_keys, internal_nodes) = collect(zip(internal_keys_and_nodes...))
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
has_leaf_node(trie::DynamicChoiceTrie, addr::Pair) = _has_leaf_node(trie, addr)
get_leaf_node(trie::DynamicChoiceTrie, addr::Pair) = _get_leaf_node(trie, addr)
has_internal_node(trie::DynamicChoiceTrie, addr::Pair) = _has_internal_node(choices, addr)
get_internal_node(trie::DynamicChoiceTrie, addr::Pair) = _get_internal_node(choices, addr)

function Base.values(trie::DynamicChoiceTrie)
    iterators::Vector = collect(map(values, trie.internal_nodes))
    push!(iterators, values(trie.leaf_nodes))
    Iterators.flatten(iterators)
end

function has_internal_node(trie::DynamicChoiceTrie, addr)
    haskey(trie.internal_nodes, addr)
end

function get_internal_node(trie::DynamicChoiceTrie, addr)
    trie.internal_nodes[addr]
end

function has_leaf_node(trie::DynamicChoiceTrie, addr)
    haskey(trie.leaf_nodes, addr)
end

function get_leaf_node(trie::DynamicChoiceTrie, addr)
    trie.leaf_nodes[addr]
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

has_leaf_node(trie::InternalVectorChoiceTrie, addr::Pair) = _has_leaf_node(trie, addr)
get_leaf_node(trie::InternalVectorChoiceTrie, addr::Pair) = _get_leaf_node(trie, addr)
has_internal_node(trie::InternalVectorChoiceTrie, addr::Pair) = _has_internal_node(trie, addr)
get_internal_node(trie::InternalVectorChoiceTrie, addr::Pair) = _get_internal_node(trie, addr)

function has_internal_node(choices::InternalVectorChoiceTrie, addr::Int)
    n = length(choices.internal_nodes)
    addr >= 1 && addr <= n
end

function get_internal_node(choices::InternalVectorChoiceTrie, addr::Int)
    choices.internal_nodes[addr]
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
