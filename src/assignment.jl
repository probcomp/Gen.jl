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
EmptyAssignment.
"""
function get_subassmt end

"""
    value = get_value(assmt::Assignment, addr)

Return the value at the given address in the assignment, or throw a KeyError if
no value exists.
"""
function get_value end

"""
    key_subassmt_iterable = get_subassmts_shallow(assmt::Assignment)

Return an iterable collection of tuples (key, subassmt) for each top-level key
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

Return an iterable collection of tuples (key, subassmt) for each top-level key
associated with a value.
"""
function get_values_shallow end

"""
In addition to the methods above, `Assignment`s implement the following:


    Base.isempty(assignment)

Are there any primitive random choice anywhere in the hierarchy?


    Base.haskey(assignment, addr)

Alias for has_value


    Base.getindex(assignment, addr)

Alias for get_value


    merge(assignment1, assignment2)

Merge two assignments.
"""
abstract type Assignment end

get_subassmt(assignment::Assignment, addr) = EmptyAssignment()
has_value(assignment::Assignment, addr) = false
get_value(assignment::Assignment, addr) = throw(KeyError(addr))
Base.getindex(assignment::Assignment, addr) = get_value(assignment, addr)

function _has_value(assmt::T, addr::Pair) where {T <: Assignment}
    (first, rest) = addr
    subassmt = get_subassmt(assmt, first)
    has_value(subassmt, rest)
end

function _get_value(assmt::T, addr::Pair) where {T <: Assignment}
    (first, rest) = addr
    subassmt = get_subassmt(assignment, first)
    get_value(subassmt, rest)
end

function _get_subassmt(assmt::T, addr::Pair) where {T <: Assignment}
    (first, rest) = addr
    subassmt = get_subassmt(assmt, first)
    get_subassmt(subassmt, rest)
end

function _print(io::IO, assignment::Assignment, pre, vert_bars::Tuple)
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
    leaf_nodes = collect(get_values_shallow(assignment))
    internal_nodes = collect(get_subassmts_shallow(assignment))
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

function Base.print(io::IO, assignment::Assignment)
    _print(io, assignment, 0, ())
end

export Assignment
export get_address_schema
export get_subassmt
export get_value
export has_value
export get_subassmts_shallow
export get_values_shallow

# assignments that have static address schemas should also support faster
# accessors, which make the address explicit in the type (Val(:foo) instaed of
# :foo)
function static_get_value end
function static_get_subassmt end
export static_get_value
export static_get_subassmt

function _fill_array! end
function _from_array end

function to_array(assmt::Assignment, ::Type{T}) where {T}
    arr = Vector{T}(undef, 32)
    n = _fill_array!(assmt, arr, 0)
    @assert n <= length(arr)
    resize!(arr, n)
    arr
end

function from_array(proto_assmt::Assignment, arr)
    (n, assmt) = _from_array(proto_assmt, arr, 0)
    if n != length(arr)
        error("Dimension mismatch: $n, $(length(arr))")
    end
    assmt
end

export to_array, from_array

function Base.merge(assignment1::Assignment, assignment2::Assignment)
    assignment = DynamicAssignment()
    for (key, value) in get_values_shallow(assignment1)
        assignment.leaf_nodes[key] = value
    end
    for (key, node1) in get_subassmts_shallow(assignment1)
        if has_internal_node(assignment2, key)
            node2 = get_subassmt(assignment2, key)
            node = merge(node1, node2)
        else
            node = node1
        end
        assignment.internal_nodes[key] = node
    end
    for (key, value) in get_values_shallow(assignment2)
        if haskey(assignment.leaf_nodes, key)
            error("assignment1 has leaf node at $key and assignment2 has leaf node at $key")
        end
        if haskey(assignment.internal_nodes, key)
            error("assignment1 has internal node at $key and assignment2 has leaf node at $key")
        end
        assignment.leaf_nodes[key] = value
    end
    for (key, node) in get_subassmts_shallow(assignment2)
        if haskey(assignment.leaf_nodes, key)
            error("assignment1 has leaf node at $key and assignment2 has internal node at $key")
        end
        if !haskey(assignment.internal_nodes, key)
            # otherwise it should already be included
            assignment.internal_nodes[key] = node
        end
    end
    return assignment
end

function Base.values(assmt::Assignment)
    iterators::Vector = collect(map(values, get_subassmts_shallow(assmt)))
    push!(iterators, values(get_values_shallow(assmt)))
    Iterators.flatten(iterators)
end


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
function Base.isempty(assignment::StaticAssignment)
    length(assignment.leaf_nodes) == 0 && length(assignment.internal_nodes) == 0
end

get_values_shallow(assignment::StaticAssignment) = pairs(assignment.leaf_nodes)
get_subassmts_shallow(assignment::StaticAssignment) = pairs(assignment.internal_nodes)
has_value(assignment::StaticAssignment, addr::Pair) = _has_value(assignment, addr)
get_value(assignment::StaticAssignment, addr::Pair) = _get_value(assignment, addr)
get_subassmt(assignment::StaticAssignment, addr::Pair) = _get_subassmt(assignment, addr)

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

function pair(a, b, key1::Symbol, key2::Symbol)
    StaticAssignment(NamedTuple(), NamedTuple{(key1,key2)}((a, b)))
end

function unpair(assmt, key1::Symbol, key2::Symbol)
    if length(get_values_shallow(assmt)) != 0 || length(get_subassmts_shallow(assmt)) > 2
        error("Not a pair")
    end
    a = get_subassmt(assmt, key1)
    b = get_subassmt(assmt, key2)
    (a, b)
end

# TODO make it a generated function?
function _fill_array!(assignment::StaticAssignment, arr::Vector{T}, start_idx::Int) where {T}
    if length(arr) < start_idx + length(assignment.leaf_nodes)
        resize!(arr, 2 * length(arr))
    end
    for (i, value) in enumerate(assignment.leaf_nodes)
        arr[start_idx + i] = value
    end
    idx = start_idx + length(assignment.leaf_nodes)
    for (i, node) in enumerate(assignment.internal_nodes)
        n_written = _fill_array!(node, arr, idx)
        idx += n_written
    end
    idx - start_idx
end

@generated function _from_array(proto_assignment::StaticAssignment{R,S,T,U}, arr::Vector{V}, start_idx::Int) where {R,S,T,U,V}
    leaf_node_keys = proto_assignment.parameters[1]
    leaf_node_types = proto_assignment.parameters[2].parameters
    internal_node_keys = proto_assignment.parameters[3]
    internal_node_types = proto_assignment.parameters[4].parameters

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
            (n_read, $node::$typ) = _from_array(proto_assignment.internal_nodes.$key, arr, idx)
            idx += n_read 
        end)
    end

    quote
        n_read = $(QuoteNode(length(leaf_node_keys)))
        leaf_nodes_field = NamedTuple{R,S}(($(leaf_node_refs...),))
        idx::Int = start_idx + n_read
        $(internal_node_blocks...)
        internal_nodes_field = NamedTuple{T,U}(($(internal_node_names...),))
        assignment = StaticAssignment{R,S,T,U}(leaf_nodes_field, internal_nodes_field)
        (idx - start_idx, assignment)
    end
end

@generated function Base.merge(assignment1::StaticAssignment{R,S,T,U},
                               assignment2::StaticAssignment{W,X,Y,Z}) where {R,S,T,U,W,X,Y,Z}

    # unpack first assignment type parameters
    leaf_node_keys1 = assignment1.parameters[1]
    leaf_node_types1 = assignment1.parameters[2].parameters
    internal_node_keys1 = assignment1.parameters[3]
    internal_node_types1 = assignment1.parameters[4].parameters
    keys1 = (leaf_node_keys1..., internal_node_keys1...,)

    # unpack second assignment type parameters
    leaf_node_keys2 = assignment2.parameters[1]
    leaf_node_types2 = assignment2.parameters[2].parameters
    internal_node_keys2 = assignment2.parameters[3]
    internal_node_types2 = assignment2.parameters[4].parameters
    keys2 = (leaf_node_keys2..., internal_node_keys2...,)

    # leaf vs leaf collision is an error
    colliding_leaf_leaf_keys = intersect(leaf_node_keys1, leaf_node_keys2)
    if !isempty(colliding_leaf_leaf_keys)
        error("assignment1 and assignment2 both have leaf nodes at key(s): $colliding_leaf_leaf_keys")
    end

    # leaf vs internal collision is an error
    colliding_leaf_internal_keys = intersect(leaf_node_keys1, internal_node_keys2)
    if !isempty(colliding_leaf_internal_keys)
        error("assignment1 has leaf node and assignment2 has internal node at key(s): $colliding_leaf_internal_keys")
    end

    # internal vs leaf collision is an error
    colliding_internal_leaf_keys = intersect(internal_node_keys1, leaf_node_keys2)
    if !isempty(colliding_internal_leaf_keys)
        error("assignment1 has internal node and assignment2 has leaf node at key(s): $colliding_internal_leaf_keys")
    end

    # internal vs internal collision is not an error, recursively call merge
    colliding_internal_internal_keys = (intersect(internal_node_keys1, internal_node_keys2)...,)
    internal_node_keys1_exclusive = (setdiff(internal_node_keys1, internal_node_keys2)...,)
    internal_node_keys2_exclusive = (setdiff(internal_node_keys2, internal_node_keys1)...,)

    # leaf nodes named tuple
    leaf_node_keys = (leaf_node_keys1..., leaf_node_keys2...,)
    leaf_node_types = map(QuoteNode, (leaf_node_types1..., leaf_node_types2...,))
    leaf_node_values = Expr(:tuple,
        [Expr(:(.), :(assignment1.leaf_nodes), QuoteNode(key))
            for key in leaf_node_keys1]...,
        [Expr(:(.), :(assignment2.leaf_nodes), QuoteNode(key))
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
        [Expr(:(.), :(assignment1.internal_nodes), QuoteNode(key))
            for key in internal_node_keys1_exclusive]...,
        [Expr(:(.), :(assignment2.internal_nodes), QuoteNode(key))
            for key in internal_node_keys2_exclusive]...,
        [Expr(:call, :merge,
                Expr(:(.), :(assignment1.internal_nodes), QuoteNode(key)),
                Expr(:(.), :(assignment2.internal_nodes), QuoteNode(key)))
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

DynamicAssignment() = DynamicAssignment(Dict(), Dict())

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

function set_value!(assignment::DynamicAssignment, addr, value)
    delete!(assignment.internal_nodes, addr)
    assignment.leaf_nodes[addr] = value
end

function set_value!(assignment::DynamicAssignment, addr::Pair, value)
    (first, rest) = addr
    if haskey(assignment.leaf_nodes, first)
        # we are not writing to the address directly, so we error instead of
        # delete the existing node.
        error("Tried to create assignment at $first but there was already a value there.")
    end
    if haskey(assignment.internal_nodes, first)
        node = assignment.internal_nodes[first]
    else
        node = DynamicAssignment()
        assignment.internal_nodes[first] = node
    end
    node = assignment.internal_nodes[first]
    set_value!(node, rest, value)
end

function set_subassmt!(assignment::DynamicAssignment, addr, new_node)
    delete!(assignment.leaf_nodes, addr)
    delete!(assignment.internal_nodes, addr)
    if !isempty(new_node)
        assignment.internal_nodes[addr] = new_node
    end
end

function set_subassmt!(assignment::DynamicAssignment, addr::Pair, new_node)
    (first, rest) = addr
    if haskey(assignment.leaf_nodes, first)
        # we are not writing to the address directly, so we error instead of
        # delete the existing node.
        error("Tried to create assignment at $first but there was already a value there.")
    end
    if haskey(assignment.internal_nodes, first)
        node = assignment.internal_nodes[first]
    else
        node = DynamicAssignment()
        assignment.internal_nodes[first] = node
    end
    set_subassmt!(node, rest, new_node)
end

Base.setindex!(assignment::DynamicAssignment, value, addr) = set_value!(assignment, addr, value)

# b should implement the assignment interface
function Base.merge!(a::DynamicAssignment, b)
    for (key, value) in get_values_shallow(b)
        a.leaf_nodes[key] = value
    end
    for (key, b_node) in get_subassmts_shallow(b)
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

function _fill_array!(assignment::DynamicAssignment, arr::Vector{T}, start_idx::Int) where {T}
    if length(arr) < start_idx + length(assignment.leaf_nodes)
        resize!(arr, 2 * length(arr))
    end
    leaf_keys_sorted = sort(collect(keys(assignment.leaf_nodes)))
    internal_node_keys_sorted = sort(collect(keys(assignment.internal_nodes)))
    for (i, key) in enumerate(leaf_keys_sorted)
        arr[start_idx + i] = assignment.leaf_nodes[key]
    end
    idx = start_idx + length(assignment.leaf_nodes)
    for key in internal_node_keys_sorted
        n_written = _fill_array!(get_subassmt(assignment, key), arr, idx)
        idx += n_written
    end
    idx - start_idx
end

function _from_array(proto_assignment::DynamicAssignment, arr::Vector{T}, start_idx::Int) where {T}
    @assert length(arr) >= start_idx
    assignment = DynamicAssignment()
    leaf_keys_sorted = sort(collect(keys(proto_assignment.leaf_nodes)))
    internal_node_keys_sorted = sort(collect(keys(proto_assignment.internal_nodes)))
    for (i, key) in enumerate(leaf_keys_sorted)
        assignment.leaf_nodes[key] = arr[start_idx + i]
    end
    idx = start_idx + length(assignment.leaf_nodes)
    for key in internal_node_keys_sorted
        (n_read, node) = _from_array(get_subassmt(proto_assignment, key), arr, idx)
        idx += n_read
        assignment.internal_nodes[key] = node
    end
    (idx - start_idx, assignment)
end

export DynamicAssignment


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
has_internal_node(assmt::InternalVectorAssignment, addr::Pair) = _has_internal_node(assmt, addr)
get_subassmt(assmt::InternalVectorAssignment, addr::Pair) = _get_subassmt(assmt, addr)

function has_internal_node(assmt::InternalVectorAssignment, addr::Int)
    n = length(assmt.internal_nodes)
    addr >= 1 && addr <= n
end

function get_subassmt(assmt::InternalVectorAssignment, addr::Int)
    if haskey(assmt.internal_nodes, addr)
        assmt.internal_nodes[addr]
    else
        EmptyAssignment()
    end
end

function get_subassmts_shallow(assmt::InternalVectorAssignment)
    ((i, subassmt)
     for subassmt in assmt.internal_nodes
     if !isempty(subassmt))
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
