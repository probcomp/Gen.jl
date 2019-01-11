#########################
# assignment interface #
#########################

"""
    schema = get_address_schema(::Type{T}) where {T <: Assignment}

Return the (top-level) address schema for the given assignment.
"""
function get_address_schema end

"""
    subassmt = get_subassmt(assmt::Assignment, addr)

Return the sub-assignment containing all choices whose address is prefixed by addr.

It is an error if the assignment contains a value at the given address. If
there are no choices whose address is prefixed by addr then return an
`EmptyAssignment`.
"""
function get_subassmt end

"""
    value = get_value(assmt::Assignment, addr)

Return the value at the given address in the assignment, or throw a KeyError if
no value exists. A syntactic sugar is `Base.getindex`:

    value = assmt[addr]
"""
function get_value end

"""
    key_subassmt_iterable = get_subassmts_shallow(assmt::Assignment)

Return an iterable collection of tuples `(key, subassmt::Assignment)` for each top-level key
that has a non-empty sub-assignment.
"""
function get_subassmts_shallow end

"""
    has_value(assmt::Assignment, addr)

Return true if there is a value at the given address.
"""
function has_value end

"""
    key_subassmt_iterable = get_values_shallow(assmt::Assignment)

Return an iterable collection of tuples `(key, value)` for each
top-level key associated with a value.
"""
function get_values_shallow end

abstract type Assignment end

"""
    Base.isempty(assmt::Assignment)

Return true if there are no values in the assignment.
"""
function Base.isempty(::Assignment)
    true
end

get_subassmt(assmt::Assignment, addr) = EmptyAssignment()
has_value(assmt::Assignment, addr) = false
get_value(assmt::Assignment, addr) = throw(KeyError(addr))
Base.getindex(assmt::Assignment, addr) = get_value(assmt, addr)

function _has_value(assmt::T, addr::Pair) where {T <: Assignment}
    (first, rest) = addr
    subassmt = get_subassmt(assmt, first)
    has_value(subassmt, rest)
end

function _get_value(assmt::T, addr::Pair) where {T <: Assignment}
    (first, rest) = addr
    subassmt = get_subassmt(assmt, first)
    get_value(subassmt, rest)
end

function _get_subassmt(assmt::T, addr::Pair) where {T <: Assignment}
    (first, rest) = addr
    subassmt = get_subassmt(assmt, first)
    get_subassmt(subassmt, rest)
end

function _print(io::IO, assmt::Assignment, pre, vert_bars::Tuple)
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
    key_and_values = collect(get_values_shallow(assmt))
    key_and_subassmts = collect(get_subassmts_shallow(assmt))
    n = length(key_and_values) + length(key_and_subassmts)
    cur = 1
    for (key, value) in key_and_values
        print(io, indent_vert_str)
        print(io, (cur == n ? indent_last_str : indent_str) * "$(repr(key)) : $value\n")
        cur += 1
    end
    for (key, subassmt) in key_and_subassmts
        print(io, indent_vert_str)
        print(io, (cur == n ? indent_last_str : indent_str) * "$(repr(key))\n")
        _print(io, subassmt, pre + 4, cur == n ? (vert_bars...,) : (vert_bars..., pre+1))
        cur += 1
    end
end

function Base.print(io::IO, assmt::Assignment)
    _print(io, assmt, 0, ())
end

# assignments that have static address schemas should also support faster
# accessors, which make the address explicit in the type (Val(:foo) instaed of
# :foo)
function static_get_value end
function static_get_subassmt end

function _fill_array! end
function _from_array end

"""
    arr::Vector{T} = to_array(assmt::Assignment, ::Type{T}) where {T}

Populate an array with values of choices in the given assignment.

It is an error if each of the values cannot be coerced into a value of the
given type.

# Implementation

To support `to_array`, a concrete subtype `T <: Assignment` should implement
the following method:

    n::Int = _fill_array!(assmt::T, arr::Vector{V}, start_idx::Int) where {V}

Populate `arr` with values from the given assignment, starting at `start_idx`,
and return the number of elements in `arr` that were populated.
"""
function to_array(assmt::Assignment, ::Type{T}) where {T}
    arr = Vector{T}(undef, 32)
    n = _fill_array!(assmt, arr, 0)
    @assert n <= length(arr)
    resize!(arr, n)
    arr
end

"""
    assmt::Assignment = from_array(proto_assmt::Assignment, arr::Vector)

Return an assignment with the same address structure as a prototype
assignment, but with values read off from the given array.

The order in which addresses are populated is determined by the prototype
assignment. It is an error if the number of choices in the prototype assignment
is not equal to the length the array.

# Implementation

To support `from_array`, a concrete subtype `T <: Assignment` should implement
the following method:


    (n::Int, assmt::T) = _from_array(proto_assmt::T, arr::Vector{V}, start_idx::Int) where {V}

Return an assignment with the same address structure as a prototype assignment,
but with values read off from `arr`, starting at position `start_idx`, and the
number of elements read from `arr`.
"""
function from_array(proto_assmt::Assignment, arr::Vector)
    (n, assmt) = _from_array(proto_assmt, arr, 0)
    if n != length(arr)
        error("Dimension mismatch: $n, $(length(arr))")
    end
    assmt
end


"""
    assmt = Base.merge(assmt1::Assignment, assmt2::Assignment)

Merge two assignments.

It is an error if the assignments both have values at the same address, or if
one assignment has a value at an address that is the prefix of the address of a
value in the other assignment.
"""
function Base.merge(assmt1::Assignment, assmt2::Assignment)
    assmt = DynamicAssignment()
    for (key, value) in get_values_shallow(assmt1)
        assmt.leaf_nodes[key] = value
    end
    for (key, node1) in get_subassmts_shallow(assmt1)
        node2 = get_subassmt(assmt2, key)
        node = merge(node1, node2)
        assmt.internal_nodes[key] = node
    end
    for (key, value) in get_values_shallow(assmt2)
        if haskey(assmt.leaf_nodes, key)
            error("assmt1 has leaf node at $key and assmt2 has leaf node at $key")
        end
        if haskey(assmt.internal_nodes, key)
            error("assmt1 has internal node at $key and assmt2 has leaf node at $key")
        end
        assmt.leaf_nodes[key] = value
    end
    for (key, node) in get_subassmts_shallow(assmt2)
        if haskey(assmt.leaf_nodes, key)
            error("assmt1 has leaf node at $key and assmt2 has internal node at $key")
        end
        if !haskey(assmt.internal_nodes, key)
            # otherwise it should already be included
            assmt.internal_nodes[key] = node
        end
    end
    return assmt
end

"""
    addrs::AddressSet = address_set(assmt::Assignment)

Return an `AddressSet` containing the addresses of values in the given assignment.
"""
function address_set(assmt::Assignment)
    set = DynamicAddressSet()
    for (key, _) in get_values_shallow(assmt)
        push_leaf_node!(set, key)
    end
    for (key, subassmt) in get_subassmts_shallow(assmt)
        set_internal_node!(set, key, address_set(subassmt))
    end
    set
end

export Assignment
export get_address_schema
export get_subassmt
export get_value
export has_value
export get_subassmts_shallow
export get_values_shallow
export static_get_value
export static_get_subassmt
export to_array, from_array
export address_set


######################
# static assignment #
######################

struct StaticAssignment{R,S,T,U} <: Assignment
    leaf_nodes::NamedTuple{R,S}
    internal_nodes::NamedTuple{T,U}
end

# TODO invariant: all internal_nodes are nonempty, but this is not verified at construction time

function get_address_schema(::Type{StaticAssignment{R,S,T,U}}) where {R,S,T,U}
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

# invariant: intenral nodes are nonempty?
function Base.isempty(assmt::StaticAssignment)
    length(assmt.leaf_nodes) == 0 && length(assmt.internal_nodes) == 0
end

get_values_shallow(assmt::StaticAssignment) = pairs(assmt.leaf_nodes)
get_subassmts_shallow(assmt::StaticAssignment) = pairs(assmt.internal_nodes)
has_value(assmt::StaticAssignment, addr::Pair) = _has_value(assmt, addr)
get_value(assmt::StaticAssignment, addr::Pair) = _get_value(assmt, addr)
get_subassmt(assmt::StaticAssignment, addr::Pair) = _get_subassmt(assmt, addr)

# NOTE: there is no static_has_value because this is known from the static
# address schema

## has_value ##

function has_value(assmt::StaticAssignment, key::Symbol)
    haskey(assmt.leaf_nodes, key)
end

## get_subassmt ##

function get_subassmt(assmt::StaticAssignment, key::Symbol)
    if haskey(assmt.internal_nodes, key)
        assmt.internal_nodes[key]
    elseif haskey(assmt.leaf_nodes, key)
        throw(KeyError(key))
    else
        EmptyAssignment()
    end
end

function static_get_subassmt(assmt::StaticAssignment, ::Val{A}) where {A}
    assmt.internal_nodes[A]
end

## get_value ##

function get_value(assmt::StaticAssignment, key::Symbol)
    assmt.leaf_nodes[key]
end

function static_get_value(assmt::StaticAssignment, ::Val{A}) where {A}
    assmt.leaf_nodes[A]
end

# convert from any other schema that has only Val{:foo} addresses
function StaticAssignment(other::Assignment)
    leaf_keys_and_nodes = collect(get_values_shallow(other))
    internal_keys_and_nodes = collect(get_subassmts_shallow(other))
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
    StaticAssignment(
        NamedTuple{leaf_keys}(leaf_nodes),
        NamedTuple{internal_keys}(internal_nodes))
end

"""
    assmt = pair(assmt1::Assignment, assmt2::Assignment, key1::Symbol, key2::Symbol)

Return an assignment that contains `assmt1` as a sub-assignment under `key1`
and `assmt2` as a sub-assignment under `key2`.
"""
function pair(assmt1::Assignment, assmt2::Assignment, key1::Symbol, key2::Symbol)
    StaticAssignment(NamedTuple(), NamedTuple{(key1,key2)}((assmt1, assmt2)))
end

"""
    (assmt1, assmt2) = unpair(assmt::Assignment, key1::Symbol, key2::Symbol)

Return the two sub-assignments at `key1` and `key2`, one or both of which may be empty.

It is an error if there are any top-level values, or any non-empty top-level
sub-assignments at keys other than `key1` and `key2`.
"""
function unpair(assmt::Assignment, key1::Symbol, key2::Symbol)
    if !isempty(get_values_shallow(assmt)) || length(collect(get_subassmts_shallow(assmt))) > 2
        error("Not a pair")
    end
    a = get_subassmt(assmt, key1)
    b = get_subassmt(assmt, key2)
    (a, b)
end

# TODO make it a generated function?
function _fill_array!(assmt::StaticAssignment, arr::Vector{T}, start_idx::Int) where {T}
    if length(arr) < start_idx + length(assmt.leaf_nodes)
        resize!(arr, 2 * length(arr))
    end
    for (i, value) in enumerate(assmt.leaf_nodes)
        arr[start_idx + i] = value
    end
    idx = start_idx + length(assmt.leaf_nodes)
    for (i, node) in enumerate(assmt.internal_nodes)
        n_written = _fill_array!(node, arr, idx)
        idx += n_written
    end
    idx - start_idx
end

@generated function _from_array(proto_assmt::StaticAssignment{R,S,T,U}, arr::Vector{V}, start_idx::Int) where {R,S,T,U,V}
    leaf_node_keys = proto_assmt.parameters[1]
    leaf_node_types = proto_assmt.parameters[2].parameters
    internal_node_keys = proto_assmt.parameters[3]
    internal_node_types = proto_assmt.parameters[4].parameters

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
            (n_read, $node::$typ) = _from_array(proto_assmt.internal_nodes.$key, arr, idx)
            idx += n_read 
        end)
    end

    quote
        n_read = $(QuoteNode(length(leaf_node_keys)))
        leaf_nodes_field = NamedTuple{R,S}(($(leaf_node_refs...),))
        idx::Int = start_idx + n_read
        $(internal_node_blocks...)
        internal_nodes_field = NamedTuple{T,U}(($(internal_node_names...),))
        assmt = StaticAssignment{R,S,T,U}(leaf_nodes_field, internal_nodes_field)
        (idx - start_idx, assmt)
    end
end

@generated function Base.merge(assmt1::StaticAssignment{R,S,T,U},
                               assmt2::StaticAssignment{W,X,Y,Z}) where {R,S,T,U,W,X,Y,Z}

    # unpack first assignment type parameters
    leaf_node_keys1 = assmt1.parameters[1]
    leaf_node_types1 = assmt1.parameters[2].parameters
    internal_node_keys1 = assmt1.parameters[3]
    internal_node_types1 = assmt1.parameters[4].parameters
    keys1 = (leaf_node_keys1..., internal_node_keys1...,)

    # unpack second assignment type parameters
    leaf_node_keys2 = assmt2.parameters[1]
    leaf_node_types2 = assmt2.parameters[2].parameters
    internal_node_keys2 = assmt2.parameters[3]
    internal_node_types2 = assmt2.parameters[4].parameters
    keys2 = (leaf_node_keys2..., internal_node_keys2...,)

    # leaf vs leaf collision is an error
    colliding_leaf_leaf_keys = intersect(leaf_node_keys1, leaf_node_keys2)
    if !isempty(colliding_leaf_leaf_keys)
        error("assmt1 and assmt2 both have leaf nodes at key(s): $colliding_leaf_leaf_keys")
    end

    # leaf vs internal collision is an error
    colliding_leaf_internal_keys = intersect(leaf_node_keys1, internal_node_keys2)
    if !isempty(colliding_leaf_internal_keys)
        error("assmt1 has leaf node and assmt2 has internal node at key(s): $colliding_leaf_internal_keys")
    end

    # internal vs leaf collision is an error
    colliding_internal_leaf_keys = intersect(internal_node_keys1, leaf_node_keys2)
    if !isempty(colliding_internal_leaf_keys)
        error("assmt1 has internal node and assmt2 has leaf node at key(s): $colliding_internal_leaf_keys")
    end

    # internal vs internal collision is not an error, recursively call merge
    colliding_internal_internal_keys = (intersect(internal_node_keys1, internal_node_keys2)...,)
    internal_node_keys1_exclusive = (setdiff(internal_node_keys1, internal_node_keys2)...,)
    internal_node_keys2_exclusive = (setdiff(internal_node_keys2, internal_node_keys1)...,)

    # leaf nodes named tuple
    leaf_node_keys = (leaf_node_keys1..., leaf_node_keys2...,)
    leaf_node_types = map(QuoteNode, (leaf_node_types1..., leaf_node_types2...,))
    leaf_node_values = Expr(:tuple,
        [Expr(:(.), :(assmt1.leaf_nodes), QuoteNode(key))
            for key in leaf_node_keys1]...,
        [Expr(:(.), :(assmt2.leaf_nodes), QuoteNode(key))
            for key in leaf_node_keys2]...)
    leaf_nodes = Expr(:call,
        Expr(:curly, :NamedTuple,
            QuoteNode(leaf_node_keys),
            Expr(:curly, :Tuple, leaf_node_types...)),
        leaf_node_values)

    # internal nodes named tuple
    internal_node_keys = (internal_node_keys1_exclusive...,
                          internal_node_keys2_exclusive...,
                          colliding_internal_internal_keys...)
    internal_node_values = Expr(:tuple,
        [Expr(:(.), :(assmt1.internal_nodes), QuoteNode(key))
            for key in internal_node_keys1_exclusive]...,
        [Expr(:(.), :(assmt2.internal_nodes), QuoteNode(key))
            for key in internal_node_keys2_exclusive]...,
        [Expr(:call, :merge,
                Expr(:(.), :(assmt1.internal_nodes), QuoteNode(key)),
                Expr(:(.), :(assmt2.internal_nodes), QuoteNode(key)))
            for key in colliding_internal_internal_keys]...)
    internal_nodes = Expr(:call,
        Expr(:curly, :NamedTuple, QuoteNode(internal_node_keys)),
        internal_node_values)

    # construct assignment from named tuples
    Expr(:call, :StaticAssignment, leaf_nodes, internal_nodes)
end

export StaticAssignment
export pair, unpair

#######################
# dynamic assignment #
#######################

struct DynamicAssignment <: Assignment
    leaf_nodes::Dict{Any,Any}
    internal_nodes::Dict{Any,Any}
end

# invariant: all internal nodes are nonempty

"""
    assmt = DynamicAssignment()

Construct an empty dynamic assignment.
"""
function DynamicAssignment()
    DynamicAssignment(Dict(), Dict())
end

get_address_schema(::Type{DynamicAssignment}) = DynamicAddressSchema()

get_values_shallow(assmt::DynamicAssignment) = assmt.leaf_nodes

get_subassmts_shallow(assmt::DynamicAssignment) = assmt.internal_nodes

has_value(assmt::DynamicAssignment, addr::Pair) = _has_value(assmt, addr)

get_value(assmt::DynamicAssignment, addr::Pair) = _get_value(assmt, addr)

get_subassmt(assmt::DynamicAssignment, addr::Pair) = _get_subassmt(assmt, addr)

function get_subassmt(assmt::DynamicAssignment, addr)
    if haskey(assmt.internal_nodes, addr)
        assmt.internal_nodes[addr]
    elseif haskey(assmt.leaf_nodes, addr)
        throw(KeyError(addr))
    else
        EmptyAssignment()
    end
end

has_value(assmt::DynamicAssignment, addr) = haskey(assmt.leaf_nodes, addr)

get_value(assmt::DynamicAssignment, addr) = assmt.leaf_nodes[addr]

function Base.isempty(assmt::DynamicAssignment)
    isempty(assmt.leaf_nodes) && isempty(assmt.internal_nodes)
end

# mutation (not part of the assignment interface)

"""
    set_value!(assmt::DynamicAssignment, addr, value)

Set the given value for the given address.

Will cause any previous value or sub-assignment at this address to be deleted.
It is an error if there is already a value present at some prefix of the given address.

The following syntactic sugar is provided:

    assmt[addr] = value
"""
function set_value!(assmt::DynamicAssignment, addr, value)
    delete!(assmt.internal_nodes, addr)
    assmt.leaf_nodes[addr] = value
end

function set_value!(assmt::DynamicAssignment, addr::Pair, value)
    (first, rest) = addr
    if haskey(assmt.leaf_nodes, first)
        # we are not writing to the address directly, so we error instead of
        # delete the existing node.
        error("Tried to create assignment at $first but there was already a value there.")
    end
    if haskey(assmt.internal_nodes, first)
        node = assmt.internal_nodes[first]
    else
        node = DynamicAssignment()
        assmt.internal_nodes[first] = node
    end
    node = assmt.internal_nodes[first]
    set_value!(node, rest, value)
end

"""
    set_subassmt!(assmt::DynamicAssignment, addr, subassmt::Assignment)

Replace the sub-assignment rooted at the given address with the given sub-assignment.
Set the given value for the given address.

Will cause any previous value or sub-assignment at the given address to be deleted.
It is an error if there is already a value present at some prefix of address.
"""
function set_subassmt!(assmt::DynamicAssignment, addr, new_node)
    delete!(assmt.leaf_nodes, addr)
    delete!(assmt.internal_nodes, addr)
    if !isempty(new_node)
        assmt.internal_nodes[addr] = new_node
    end
end

function set_subassmt!(assmt::DynamicAssignment, addr::Pair, new_node)
    (first, rest) = addr
    if haskey(assmt.leaf_nodes, first)
        # we are not writing to the address directly, so we error instead of
        # delete the existing node.
        error("Tried to create assignment at $first but there was already a value there.")
    end
    if haskey(assmt.internal_nodes, first)
        node = assmt.internal_nodes[first]
    else
        node = DynamicAssignment()
        assmt.internal_nodes[first] = node
    end
    set_subassmt!(node, rest, new_node)
end

Base.setindex!(assmt::DynamicAssignment, value, addr) = set_value!(assmt, addr, value)

function _fill_array!(assmt::DynamicAssignment, arr::Vector{T}, start_idx::Int) where {T}
    if length(arr) < start_idx + length(assmt.leaf_nodes)
        resize!(arr, 2 * length(arr))
    end
    leaf_keys_sorted = sort(collect(keys(assmt.leaf_nodes)))
    internal_node_keys_sorted = sort(collect(keys(assmt.internal_nodes)))
    for (i, key) in enumerate(leaf_keys_sorted)
        arr[start_idx + i] = assmt.leaf_nodes[key]
    end
    idx = start_idx + length(assmt.leaf_nodes)
    for key in internal_node_keys_sorted
        n_written = _fill_array!(get_subassmt(assmt, key), arr, idx)
        idx += n_written
    end
    idx - start_idx
end

function _from_array(proto_assmt::DynamicAssignment, arr::Vector{T}, start_idx::Int) where {T}
    @assert length(arr) >= start_idx
    assmt = DynamicAssignment()
    leaf_keys_sorted = sort(collect(keys(proto_assmt.leaf_nodes)))
    internal_node_keys_sorted = sort(collect(keys(proto_assmt.internal_nodes)))
    for (i, key) in enumerate(leaf_keys_sorted)
        assmt.leaf_nodes[key] = arr[start_idx + i]
    end
    idx = start_idx + length(assmt.leaf_nodes)
    for key in internal_node_keys_sorted
        (n_read, node) = _from_array(get_subassmt(proto_assmt, key), arr, idx)
        idx += n_read
        assmt.internal_nodes[key] = node
    end
    (idx - start_idx, assmt)
end

export DynamicAssignment
export set_value!
export set_subassmt!


#######################################
## vector combinator for assignments #
#######################################

# TODO implement LeafVectorAssignment, which stores a vector of leaf nodes

struct InternalVectorAssignment{T} <: Assignment
    internal_nodes::Vector{T}
    is_empty::Bool
end

function vectorize_internal(nodes::Vector{T}) where {T}
    is_empty = all(map(isempty, nodes))
    InternalVectorAssignment(nodes, is_empty)
end

# note some internal nodes may be empty

get_address_schema(::Type{InternalVectorAssignment}) = VectorAddressSchema()

Base.isempty(assmt::InternalVectorAssignment) = assmt.is_empty
has_value(assmt::InternalVectorAssignment, addr::Pair) = _has_value(assmt, addr)
get_value(assmt::InternalVectorAssignment, addr::Pair) = _get_value(assmt, addr)
get_subassmt(assmt::InternalVectorAssignment, addr::Pair) = _get_subassmt(assmt, addr)

function get_subassmt(assmt::InternalVectorAssignment, addr::Int)
    if addr > 0 && addr <= length(assmt.internal_nodes)
        assmt.internal_nodes[addr]
    else
        EmptyAssignment()
    end
end

function get_subassmts_shallow(assmt::InternalVectorAssignment)
    ((i, assmt.internal_nodes[i])
     for i=1:length(assmt.internal_nodes)
     if !isempty(assmt.internal_nodes[i]))
end

get_values_shallow(::InternalVectorAssignment) = ()

function _fill_array!(assmt::InternalVectorAssignment, arr::Vector{T}, start_idx::Int) where {T}
    idx = start_idx
    for key=1:length(assmt.internal_nodes)
        n = _fill_array!(assmt.internal_nodes[key], arr, idx)
        idx += n
    end
    idx - start_idx
end

function _from_array(proto_assmt::InternalVectorAssignment{U}, arr::Vector{T}, start_idx::Int) where {T,U}
    @assert length(arr) >= start_idx
    nodes = Vector{U}(undef, length(proto_assmt.internal_nodes))
    idx = start_idx
    for key=1:length(proto_assmt.internal_nodes)
        (n_read, nodes[key]) = _from_array(proto_assmt.internal_nodes[key], arr, idx)
        idx += n_read
    end
    assmt = InternalVectorAssignment(nodes, proto_assmt.is_empty)
    (idx - start_idx, assmt)
end

export InternalVectorAssignment
export vectorize_internal


####################
# empty assignment #
####################

struct EmptyAssignment <: Assignment end

Base.isempty(::EmptyAssignment) = true
get_address_schema(::Type{EmptyAssignment}) = EmptyAddressSchema()
get_subassmts_shallow(::EmptyAssignment) = ()
get_values_shallow(::EmptyAssignment) = ()

_fill_array!(::EmptyAssignment, arr::Vector, start_idx::Int) = 0
_from_array(::EmptyAssignment, arr::Vector, start_idx::Int) = (0, EmptyAssignment())

export EmptyAssignment
