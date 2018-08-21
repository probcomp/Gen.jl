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


#######################
# Generic choice trie #
#######################

struct GenericChoiceTrie
    leaf_nodes::Dict{Any,Any}
    internal_nodes::Dict{Any,Any}
end

GenericChoiceTrie() = GenericChoiceTrie(Dict(), Dict())

# invariant: all internal nodes are nonempty
Base.isempty(trie::GenericChoiceTrie) = isempty(trie.leaf_nodes) && isempty(trie.internal_nodes)

get_leaf_nodes(trie::GenericChoiceTrie) = trie.leaf_nodes

get_internal_nodes(trie::GenericChoiceTrie) = trie.internal_nodes

function Base.values(trie::GenericChoiceTrie)
    iterators::Vector = collect(map(values, trie.internal_nodes))
    push!(iterators, values(trie.leaf_nodes))
    Iterators.flatten(iterators)
end

function has_internal_node(trie::GenericChoiceTrie, addr)
    haskey(trie.internal_nodes, addr)
end

function has_internal_node(trie::GenericChoiceTrie, addr::Pair)
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        node = trie.internal_nodes[first]
        has_internal_node(node, rest)
    else
        false
    end
end

function get_internal_node(trie::GenericChoiceTrie, addr)
    trie.internal_nodes[addr]
end

function get_internal_node(trie::GenericChoiceTrie, addr::Pair)
    (first, rest) = addr
    node = trie.internal_nodes[first]
    get_internal_node(node, rest)
end

function set_internal_node!(trie::GenericChoiceTrie, addr, new_node)
    if !isempty(new_node)
        if haskey(trie.internal_nodes, addr)
            error("Node already exists at $addr")
        end
        trie.internal_nodes[addr] = new_node
    end
end

function set_internal_node!(trie::GenericChoiceTrie, addr::Pair, new_node)
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        node = trie.internal_nodes[first]
    else
        node = GenericChoiceTrie()
        trie.internal_nodes[first] = node
    end
    set_internal_node!(node, rest, new_node)
end

function has_leaf_node(trie::GenericChoiceTrie, addr)
    haskey(trie.leaf_nodes, addr)
end

function has_leaf_node(trie::GenericChoiceTrie, addr::Pair)
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        node = trie.internal_nodes[first]
        has_leaf_node(node, rest)
    else
        false
    end
end

function get_leaf_node(trie::GenericChoiceTrie, addr)
    trie.leaf_nodes[addr]
end

function get_leaf_node(trie::GenericChoiceTrie, addr::Pair)
    (first, rest) = addr
    node = trie.internal_nodes[first]
    get_leaf_node(node, rest)
end

function set_leaf_node!(trie::GenericChoiceTrie, addr, value)
    trie.leaf_nodes[addr] = value
end

function set_leaf_node!(trie::GenericChoiceTrie, addr::Pair, value)
    (first, rest) = addr
    if haskey(trie.internal_nodes, first)
        node = trie.internal_nodes[first]
    else
        node = GenericChoiceTrie()
        trie.internal_nodes[first] = node
    end
    node = trie.internal_nodes[first]
    set_leaf_node!(node, rest, value)
end

Base.setindex!(trie::GenericChoiceTrie, value, addr) = set_leaf_node!(trie, addr, value)
Base.haskey(trie::GenericChoiceTrie, addr) = has_leaf_node(trie, addr)
Base.getindex(trie::GenericChoiceTrie, addr) = get_leaf_node(trie, addr)

# b should implement the choice-trie interface
function Base.merge!(a::GenericChoiceTrie, b)
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

get_address_schema(::Type{GenericChoiceTrie}) = DynamicAddressSchema()
get_internal_node_proto(::GenericChoiceTrie, addr) = GenericChoiceTrie()

export GenericChoiceTrie


####################################
# SchemaAnnotatedGenericChoiceTrie #
####################################

#abstract type WrappedGenericChoiceTrie end
#
#struct StaticAddressSchemaChoiceTrie{T} # T will be a Tuple of Val{:symbol} types
    #trie::GenericChoiceTrie
#end
#
#function get_address_schema(::Type{StaticAddressSchemaChoiceTrie{T,U}}) where {T <: Tuple, U <: Tuple}
    ## T are the primitive random choice addresses, U are the subtrace addresses
    #fields = Dict{Symbol,StaticAddressInfo}()
    #for val_field in T.parameters
        #addr::Symbol = val_field.parameters[1]
        #fields[addr] = StaticAddressInfo(true)
    #end
    #for val_field in U.parameters
        #addr::Symbol = val_field.parameters[1]
        #fields[addr] = StaticAddressInfo(false)
    #end
    #StaticAddressSchema(fields)
#end
#
#struct SingleDynamicKeyChoiceTrie
    #trie::GenericChoiceTrie
#end
#
#get_address_schema(::Type{SingleDynamicKeyChoiceTrie}) = SingleDynamicKeyAddressSchema()
#
#function wrap_with_address_schema(trie::GenericChoiceTrie)
    ## TODO have a special field in generic choice trie that checks if they are all symbols? (so we know if we are static or not)
    ## TODO have a special field that checks if there is only one address (so we know if we are single-dynamic or not)
    ## these fields are updated as the trie is constructed (during update or backprop_trace)
#end
#
## unlike the generic choice trie, it is read only
#Base.isempty(wrapped::WrappedGenericChoiceTrie) = isempty(wrapped.trie)
#get_leaf_nodes(wrapped::WrappedGenericChoiceTrie) = get_leaf_nodes(wrapped.trie)
#get_internal_nodes(wrapped::WrappedGenericChoiceTrie) = get_internal_nodes(wrapped.trie)
#Base.values(wrapped::WrappedGenericChoiceTrie) = values(wrapped.trie)
#has_internal_node(wrapped::WrappedGenericChoiceTrie, addr) = has_internal_node(wrapped.trie, addr)
#get_internal_node(wrapped::WrappedGenericChoiceTrie, addr) = get_internal_node(wrapped.trie, addr)
#has_leaf_node(wrapped::WrappedGenericChoiceTrie, addr) = has_leaf_node(wrapped.trie, addr)
#get_leaf_node(wrapped::WrappedGenericChoiceTrie, addr) = get_leaf_node(wrapped.trie, addr)
#Base.haskey(wrapped::WrappedGenericChoiceTrie, addr) = haskey(wrapped.trie, addr)
#Base.getindex(wrapped::WrappedGenericChoiceTrie, addr) = getindex(wrapped.trie, addr)

#######################################
## Vector combinator for choice tries #
#######################################

# TODO: implement another version that represents a vector of primitive choices
struct VectorChoiceTrie{T}
    subtries::Vector{T}
    is_empty::Bool
end

Base.isempty(choices::VectorChoiceTrie) = choices.is_empty

function vectorize(choices::Vector{T}) where {T}
    is_empty = all(map(isempty, choices))
    VectorChoiceTrie{T}(choices, is_empty)
end

has_internal_node(choices::VectorChoiceTrie, addr) = false

function has_internal_node(choices::VectorChoiceTrie, addr::Int)
    n = length(choices.subtries)
    addr >= 1 && addr <= n
end

function has_internal_node(choices::VectorChoiceTrie, addr::Pair)
    (first, rest) = addr
    subtrie = choices.subtries[first]
    has_internal_node(subtrie, rest)
end

function get_internal_node(choices::VectorChoiceTrie, addr::Int)
    choices.subtries[addr]
end

function get_internal_node(choices::VectorChoiceTrie, addr::Pair)
    (first, rest) = addr
    subtrie = choices.subtries[first]
    get_internal_node(subtrie, rest)
end

has_leaf_node(choices::VectorChoiceTrie, addr) = false

function has_leaf_node(choices::VectorChoiceTrie, addr::Pair)
    (first, rest) = addr
    subtrie = choices.subtries[first]
    has_leaf_node(subtrie, rest)
end

function get_leaf_node(choices::VectorChoiceTrie, addr::Pair)
    (first, rest) = addr
    subtrie = choices.subtries[first]
    get_leaf_node(subtrie, rest)
end

function get_internal_nodes(choices::VectorChoiceTrie)
    ((i, choices.subtries[i]) for i=1:length(choices.subtries))
end

get_leaf_nodes(choices::VectorChoiceTrie) = ()

Base.haskey(choices::VectorChoiceTrie, addr) = has_leaf_node(choices, addr)
Base.getindex(choices::VectorChoiceTrie, addr) = get_leaf_node(choices, addr)

# TODO create a static version that implements get_address_schema

export vectorize

####################################
# Pair combinator for choice tries #
####################################

function pair(choices1, choices2, key1, key2)
    choices = GenericChoiceTrie()
    set_internal_node!(choices, key1, choices1)
    set_internal_node!(choices, key2, choices2)
    choices
end

function unpair(choices, key1, key2)
    if length(get_internal_nodes(choices)) > 2
        error("Choice trie was not a pair")
    end
    if has_internal_node(choices, key1)
        choices1 = get_internal_node(choices, key1)
    else
        choices1 = EmptyChoiceTrie()
    end
    if has_internal_node(choices, key2)
        choices2 = get_internal_node(choices, key2)
    else
        choices2 = EmptyChoiceTrie()
    end
    (choices1, choices2)
end

# TODO create a static version that implements get_address_schema

export pair, unpair

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
