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
#

############################
# (shallow) address schema #
############################

struct StaticAddressInfo
    is_primitive::Bool
end

# will be produced for a basic block, and for if-else
# actual keys are wrapped in a value type e.g. Val(:foo) not :foo
struct StaticAddressSchema 
    #keys::Set{Symbol}
    fields::Dict{Symbol, StaticAddressInfo}
end

Base.keys(schema::StaticAddressSchema) = keys(schema.fields)
Base.getindex(schema::StaticAddressSchema, key::Symbol) = schema.fields[key]

# some unknown number of integers, will be produced for markov, iid, and plate
struct VectorAddressSchema end 

# e.g. an integer index (TODO provide the type of the key?)
struct SingleDynamicKeyAddressSchema end 

# an unknown number of unknown addresses
struct DynamicAddressSchema end 

# there are no addresses
struct EmptyAddressSchema end

##############
# AddressSet #
##############

struct AddressSet
    leaf_nodes::Set{Any}
    internal_nodes::Dict{Any,AddressSet}
end

AddressSet() = AddressSet(Set{Any}(), Dict{Any,AddressSet}())

function Base.isempty(set::AddressSet)
    isempty(set.leaf_nodes) && all(map(isempty, values(set.internal_nodes)))
end

"""
Test if an address is in the collection
"""
Base.in(addr, set::AddressSet) = (addr in set.leaf_nodes)

function Base.in(addr::Pair, set::AddressSet)
    (first, rest) = addr
    if haskey(set.internal_nodes, first)
        internal_node = set.internal_nodes[first]
        in(rest, internal_node)
    else
        false
    end
end


"""
Return the subset at the given address, or create an empty one and return it if it does not exist

No copying is performed.
"""
function Base.getindex(set::AddressSet, addr)
    if addr in set.leaf_nodes
        error("Address $addr already added; cannot get subset")
    end
    if !haskey(set.internal_nodes, addr)
        set.internal_nodes[addr] = AddressSet()
    end
    set.internal_nodes[addr]
end

function Base.getindex(set::AddressSet, addr::Pair)
    (first, rest) = addr
    if !haskey(set.internal_nodes, first)
        set.internal_nodes[first] = AddressSet()
    end
    node = set.internal_nodes[first]
    getindex(node, rest)
end


"""
Add an address to the collection
"""
function Base.push!(set::AddressSet, addr)
    if haskey(set.internal_nodes, addr)
        error("Tried to push! $addr but there is already a namespace rooted at $addr")
    end
    push!(set.leaf_nodes, addr)
end

function Base.push!(set::AddressSet, addr::Pair)
    (first, rest) = addr
    if haskey(set.internal_nodes, first)
        node = set.internal_nodes[first]
    else
        node = AddressSet()
        set.internal_nodes[first] = node
    end
    push!(node, rest)
end

get_leaf_nodes(addrs::AddressSet) = addrs.leaf_nodes
get_internal_nodes(addrs::AddressSet) = addrs.internal_nodes

export Address, AddressSet
