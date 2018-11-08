#############
# addresses #
#############
#
# Addresses are linked lists of non-Pair values, using Pair as cons.
# Symbols indicate statically known address fields. (TODO unify this)
#
# Therefore, Addresses can be conveniently constructed using Julia's => syntax
# for constructing pairs:
# 
#   :a => 2 => :d => "asfd"
#
# which gives:
#
#  Pair(:a, Pair(2, Pair(:d, "asdf")))

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


########################
# abstract address set #
########################

"""
    get_address_schema(T)

    Base.isempty(set)

Is the set empty?

Return a shallow, compile-time address schema.

    has_internal_node(set, addr)

    get_internal_node(set, addr)

Return the set of address set nodes in the given namespace (they are all nonempty?)

    get_internal_nodes(set)

Return an iterator over top-level internal nodes (pairs of keys and nodes)

    has_leaf_node(set, addr)

Is the address in the set

    get_leaf_nodes(set)

Return an iterator over top-level leaf nodes (keys)

    Base.in(set, addr)

Alias for has_leaf_node

    Base.getindex(set, addr)

Alias for get_internal_node

    Base.haskey(set, addr)

Alias for has_internal_node

"""
abstract type AddressSet end

has_internal_node(::AddressSet, addr) = false
get_internal_node(::AddressSet, addr) = throw(KeyError(addr))
has_leaf_node(::AddressSet, addr) = false
Base.in(addr, set::AddressSet) = has_leaf_node(set, addr)
Base.getindex(set::AddressSet, addr) = get_internal_node(set, addr)
Base.haskey(set::AddressSet, addr) = has_internal_node(set, addr)

export AddressSet

# has_internal_node()
#       - used in backprop_trace() OK
#       - used in regenerate() OK
# get_internal_node() [[ alias getindex, KeyError if it doesn't exist ]]
#       - used in backprop_trace() (check if there is one first, and return EmptySet o/w) OK
#       - used in regenerate() (check if there is one first, and return EmptySet o/w) OK
# has_leaf_node() [[ alias in ]]
#       - used in backprop_trace() OK
#       - used in regenerate() OK
# get_leaf_nodes() [[ return iterator over keys in this namespace ]]
# get_internal_nodes() [[ return iterator of pairs of keys and sub-address-set ]]

# for static address sets (with StaticAddressSchema schema)
# static_get_internal_node(set::StaticAddressSet, ::Val{A}) where {A}
#       - used in basic block backprop_trace, regenerate, etc.

# construction interface (for DynamicAddressSet)
# there is no construction interface for StaticAddressSet
# add_leaf_node!(set, addr) [[ alias push ]]
# set_internal_node!(set, addr, node::AddressSet) [[ alias setindex ]]

# there should by a copy constructor for StaticAddressSet from dynamic..
# later selection functions can generate static address sets directly..


#####################
# empty address set #
#####################

struct EmptyAddressSet <: AddressSet end
get_address_schema(::Type{EmptyAddressSet}) = EmptyAddressSchema()
Base.isempty(::EmptyAddressSet) = true

export EmptyAddressSet


######################
# static address set #
######################

# R is a tuple of symbols..
# T is a tuple of symbols
# U the tuple type of internal nodes
struct StaticAddressSet{R,T,U} <: AddressSet
    internal_nodes::NamedTuple{T,U}
end

function Base.isempty(set::StaticAddressSet{R,T,U}) where {R,T,U}
    length(R) == 0 && all(isempty(node) for node in set.internal_nodes)
end

function get_address_schema(::Type{StaticAddressSet{R,T,U}}) where {R,T,U}
    leaf_keys = Set{Symbol}()
    internal_keys = Set{Symbol}()
    for key in R
        push!(leaf_keys, key)
    end
    for (key, _) in zip(T, U.parameters)
        push!(internal_keys, key)
    end
    StaticAddressSchema(leaf_keys, internal_keys)
end

get_leaf_nodes(::StaticAddressSet{R,T,U}) where {R,T,U} = R

get_internal_nodes(set::StaticAddressSet) = pairs(set.internal_nodes)

function has_internal_node(set::StaticAddressSet, key::Symbol)
    haskey(set.internal_nodes, key)
end

function has_internal_node(set::StaticAddressSet, addr::Pair)
    (first, rest) = addr
    if haskey(set.internal_nodes, first)
        node = set.internal_nodes[first]
        has_internal_node(node, rest)
    else
        false
    end
end

function static_get_internal_node(set::StaticAddressSet, ::Val{A}) where {A}
    set.internal_nodes[A]
end

function get_internal_node(set::StaticAddressSet, key::Symbol)
    set.internal_nodes[key]
end

function get_internal_node(set::StaticAddressSet, addr::Pair)
    (first, rest) = addr
    node = set.internal_nodes[first]
    get_internal_node(node, rest)
end

function has_leaf_node(set::StaticAddressSet{R,T,U}, key::Symbol) where {R,T,U}
    key in R # TODO this is O(N)
end

function has_leaf_node(set::StaticAddressSet, addr::Pair)
    (first, rest) = addr
    if haskey(set.internal_nodes, first)
        node = set.internal_nodes[first]
        has_leaf_node(node, rest)
    else
        false
    end
end

function StaticAddressSet(other::AddressSet)
    leaf_keys = (get_leaf_nodes(other)...,)
    internal_keys_and_nodes = collect(get_internal_nodes(other))
    if length(internal_keys_and_nodes) > 0
        (internal_keys, internal_nodes) = collect(zip(internal_keys_and_nodes...))
    else
        (internal_keys, internal_nodes) = ((), ())
    end
    internal_types = map(typeof, internal_nodes)
    StaticAddressSet{leaf_keys,internal_keys,Tuple{internal_types...}}(
        NamedTuple{internal_keys}(internal_nodes))
end

export StaticAddressSet


#######################
# dynamic address set #
#######################

struct DynamicAddressSet <: AddressSet
    leaf_nodes::Set{Any} # set of keys
    internal_nodes::Dict{Any,AddressSet}
end

# invariant: all internal nodes are nonempty
function Base.isempty(set::DynamicAddressSet)
    isempty(set.leaf_nodes) && isempty(set.internal_nodes)
end

DynamicAddressSet() = DynamicAddressSet(Set{Any}(), Dict{Any,DynamicAddressSet}())

get_address_schema(::Type{DynamicAddressSet}) = DynamicAddressSchema()

has_leaf_node(set::DynamicAddressSet, addr) = (addr in set.leaf_nodes)

function has_leaf_node(set::DynamicAddressSet, addr::Pair)
    (first, rest) = addr
    if haskey(set.internal_nodes, first)
        internal_node = set.internal_nodes[first]
        has_leaf_node(internal_node, rest)
    else
        false
    end
end

function has_internal_node(set::DynamicAddressSet, addr)
    haskey(set.internal_nodes, addr)
end

function has_internal_node(set::DynamicAddressSet, addr::Pair)
    (first, rest) = addr
    if haskey(set.internal_nodes, first)
        node = set.internal_nodes[first]
        has_internal_node(node, rest)
    else
        false
    end
end

function get_internal_node(set::DynamicAddressSet, addr)
    set.internal_nodes[addr]
end

function get_internal_node(set::DynamicAddressSet, addr::Pair)
    (first, rest) = addr
    node = set.internal_nodes[first]
    get_internal_node(node, rest)
end

function push_leaf_node!(set::DynamicAddressSet, addr)
    if haskey(set.internal_nodes, addr)
        error("Tried to push_leaf_node! $addr but there is already a namespace rooted at $addr")
    end
    push!(set.leaf_nodes, addr)
end

function push_leaf_node!(set::DynamicAddressSet, addr::Pair)
    (first, rest) = addr
    if haskey(set.internal_nodes, first)
        node = set.internal_nodes[first]
    else
        node = DynamicAddressSet()
        set.internal_nodes[first] = node
    end
    push_leaf_node!(node, rest)
end

function set_internal_node!(set::DynamicAddressSet, addr, node)
    if isempty(node)
        return
    end
    if addr in set.leaf_nodes
        error("Tried to set_internal_node! $addr but that addres is already a leaf")
    else
        set.internal_nodes[addr] = node
    end
end

function set_internal_node!(set::DynamicAddressSet, addr::Pair, node)
    if isempty(node)
        return
    end
    (first, rest) = addr
    node = set.internal_nodes[first]
    set_internal_node!(node, rest, node)
end

get_leaf_nodes(addrs::DynamicAddressSet) = addrs.leaf_nodes

get_internal_nodes(addrs::DynamicAddressSet) = addrs.internal_nodes

Base.push!(set::DynamicAddressSet, addr) = push_leaf_node!(set, addr)

export DynamicAddressSet
export push_leaf_node!, set_internal_node!
