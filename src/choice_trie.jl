###################
# Address schemas #
###################

struct StaticAddressSchema
    # TODO can add type information
    leaf_nodes::Set{Symbol}
    internal_nodes::Set{Symbol}
end

leaf_node_keys(schema::StaticAddressSchema) = schema.leaf_nodes
internal_node_keys(schema::StaticAddressSchema) = schema.internal_nodes

struct VectorAddressSchema end 
struct SingleDynamicKeyAddressSchema end 
struct DynamicAddressSchema end 
struct EmptyAddressSchema end

export StaticAddressSchema
export VectorAddressSchema
export SingleDynamicKeyAddressSchema
export DynamicAddressSchema
export EmptyAddressSchema

#########################
# Choice trie interface #
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
function get_address_schema end
function has_internal_node end
function get_internal_node end
function get_internal_nodes end
function has_leaf_node end
function get_leaf_node end
function get_leaf_nodes end

function to_nested_dict(choice_trie)
    dict = Dict()
    for (key, value) in get_leaf_nodes(choice_trie)
        dict[key] = value
    end
    for (key, node) in get_internal_nodes(choice_trie)
        dict[key] = to_nested_dict(node)
    end
    dict
end

# NOTE: if ChoiceTrie were an abstract type, then we could just make this the
# behavior of Base.print()
using JSON
function print_choices(choices)
    JSON.print(to_nested_dict(choices), 4)
end

export get_address_schema
export has_internal_node
export get_internal_node
export get_internal_nodes
export has_leaf_node
export get_leaf_node
export get_leaf_nodes 
export print_choices


######################
# static choice trie #
######################

struct StaticChoiceTrie{R,S,T,U}
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

function has_internal_node(trie::StaticChoiceTrie, ::Val{A}) where {A}
    haskey(trie.internal_nodes, A)
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

function get_internal_node(trie::StaticChoiceTrie, ::Val{A}) where {A}
    trie.internal_nodes[A]
end

function get_internal_node(trie::StaticChoiceTrie, addr::Pair)
    (first, rest) = addr
    node = trie.internal_nodes[first]
    get_internal_node(node, rest)
end

function has_leaf_node(trie::StaticChoiceTrie, ::Val{A}) where {A}
    haskey(trie.leaf_nodes, A)
end

function has_leaf_node(trie::StaticChoiceTrie, addr::Pair)
    (first, rest) = addr
    if haskey(trie.leaf_nodes, first)
        node = trie.leaf_nodes[first]
        has_leaf_node(node, rest)
    else
        false
    end
end

function get_leaf_node(trie::StaticChoiceTrie, ::Val{A}) where {A}
    trie.leaf_nodes[A]
end

function get_leaf_node(trie::StaticChoiceTrie, addr::Pair)
    (first, rest) = addr
    node = trie.internal_nodes[first]
    get_leaf_node(node, rest)
end

# convert from any other schema that has only Val{:foo} addresses
function StaticChoiceTrie(other)
    leaf_keys_and_nodes = collect(get_leaf_nodes(other))
    internal_keys_and_nodes = collect(get_internal_nodes(other))
    if length(leaf_keys_and_nodes) > 0
        (leaf_keys, leaf_nodes) = collect(zip(leaf_keys_and_nodes...))
        leaf_keys = map((k) -> typeof(k).parameters[1]::Symbol, leaf_keys)
    else
        (leaf_keys, leaf_nodes) = ((), ())
    end
    if length(internal_keys_and_nodes) > 0
        (internal_keys, internal_nodes) = collect(zip(internal_keys_and_nodes...))
        internal_keys = map((k) -> typeof(k).parameters[1]::Symbol, internal_keys)
    else
        (internal_keys, internal_nodes) = ((), ())
    end
    StaticChoiceTrie(
        NamedTuple{leaf_keys}(leaf_nodes),
        NamedTuple{internal_keys}(internal_nodes))
end

function pair(a, b, ::Val{A}, ::Val{B}) where {A,B}
    StaticChoiceTrie(NamedTuple(), NamedTuple{(A,B)}((a, b)))
end

function unpair(trie::StaticChoiceTrie, ::Val{A}, ::Val{B}) where {A,B}
    if length(trie.leaf_nodes) != 0 || length(trie.internal_nodes) > 2
        error("Not a pair")
    end
    a = haskey(trie.internal_nodes, A) ? trie.internal_nodes[A] : EmptyChoiceTrie()
    b = haskey(trie.internal_nodes, B) ? trie.internal_nodes[B] : EmptyChoiceTrie()
    (a, b)
end

export StaticChoiceTrie
export pair, unpair


#######################
# Dynamic choice trie #
#######################

struct DynamicChoiceTrie
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

Base.haskey(trie::DynamicChoiceTrie, addr) = has_leaf_node(trie, addr)
Base.getindex(trie::DynamicChoiceTrie, addr) = get_leaf_node(trie, addr)


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


export DynamicChoiceTrie


#######################################
## Vector combinator for choice tries #
#######################################

struct InternalVectorChoiceTrie{T}
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
has_internal_node(choices::InternalVectorChoiceTrie, addr) = false

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

has_leaf_node(choices::InternalVectorChoiceTrie, addr) = false

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

Base.haskey(choices::InternalVectorChoiceTrie, addr) = has_leaf_node(choices, addr)
Base.getindex(choices::InternalVectorChoiceTrie, addr) = get_leaf_node(choices, addr)

export InternalVectorChoiceTrie
export vectorize_internal


###################
# EmptyChoiceTrie #
###################

struct EmptyChoiceTrie end

Base.isempty(::EmptyChoiceTrie) = true
get_address_schema(::Type{EmptyChoiceTrie}) = EmptyAddressSchema()
has_internal_node(::EmptyChoiceTrie, addr) = false
get_internal_node(::EmptyChoiceTrie, addr) = throw(KeyError(addr))
get_internal_nodes(::EmptyChoiceTrie) = ()
has_leaf_node(::EmptyChoiceTrie, addr) = false
get_leaf_node(::EmptyChoiceTrie, addr) = throw(KeyError(addr))
get_leaf_nodes(::EmptyChoiceTrie) = ()
Base.haskey(trie::EmptyChoiceTrie, addr) = has_leaf_node(trie, addr)
Base.getindex(trie::EmptyChoiceTrie, addr) = get_leaf_node(trie, addr)

export EmptyChoiceTrie
