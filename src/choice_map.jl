#########################
# choice map interface #
#########################

"""
    schema = get_address_schema(::Type{T}) where {T <: ChoiceMap}

Return the (top-level) address schema for the given choice map.
"""
function get_address_schema end

"""
    submap = get_submap(choices::ChoiceMap, addr)

Return the sub-assignment containing all choices whose address is prefixed by addr.

It is an error if the assignment contains a value at the given address. If
there are no choices whose address is prefixed by addr then return an
`EmptyChoiceMap`.
"""
function get_submap end

"""
    value = get_value(choices::ChoiceMap, addr)

Return the value at the given address in the assignment, or throw a KeyError if
no value exists. A syntactic sugar is `Base.getindex`:

    value = choices[addr]
"""
function get_value end

"""
    key_submap_iterable = get_submaps_shallow(choices::ChoiceMap)

Return an iterable collection of tuples `(key, submap::ChoiceMap)` for each top-level key
that has a non-empty sub-assignment.
"""
function get_submaps_shallow end

"""
    has_value(choices::ChoiceMap, addr)

Return true if there is a value at the given address.
"""
function has_value end

"""
    key_submap_iterable = get_values_shallow(choices::ChoiceMap)

Return an iterable collection of tuples `(key, value)` for each
top-level key associated with a value.
"""
function get_values_shallow end

"""
    abstract type ChoiceMap end

Abstract type for maps from hierarchical addresses to values.
"""
abstract type ChoiceMap end

"""
    Base.isempty(choices::ChoiceMap)

Return true if there are no values in the assignment.
"""
function Base.isempty(::ChoiceMap)
    true
end

@inline get_submap(choices::ChoiceMap, addr) = EmptyChoiceMap()
@inline has_value(choices::ChoiceMap, addr) = false
@inline get_value(choices::ChoiceMap, addr) = throw(KeyError(addr))
@inline Base.getindex(choices::ChoiceMap, addr) = get_value(choices, addr)

@inline function _has_value(choices::T, addr::Pair) where {T <: ChoiceMap}
    (first, rest) = addr
    submap = get_submap(choices, first)
    has_value(submap, rest)
end

@inline function _get_value(choices::T, addr::Pair) where {T <: ChoiceMap}
    (first, rest) = addr
    submap = get_submap(choices, first)
    get_value(submap, rest)
end

@inline function _get_submap(choices::T, addr::Pair) where {T <: ChoiceMap}
    (first, rest) = addr
    submap = get_submap(choices, first)
    get_submap(submap, rest)
end

function _show_pretty(io::IO, choices::ChoiceMap, pre, vert_bars::Tuple)
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
    key_and_values = collect(get_values_shallow(choices))
    key_and_submaps = collect(get_submaps_shallow(choices))
    n = length(key_and_values) + length(key_and_submaps)
    cur = 1
    for (key, value) in key_and_values
        # For strings, `print` is what we want; `Base.show` includes quote marks.
        # https://docs.julialang.org/en/v1/base/io-network/#Base.print
        print(io, indent_vert_str)
        print(io, (cur == n ? indent_last_str : indent_str) * "$(repr(key)) : $value\n")
        cur += 1
    end
    for (key, submap) in key_and_submaps
        print(io, indent_vert_str)
        print(io, (cur == n ? indent_last_str : indent_str) * "$(repr(key))\n")
        _show_pretty(io, submap, pre + 4, cur == n ? (vert_bars...,) : (vert_bars..., pre+1))
        cur += 1
    end
end

function Base.show(io::IO, ::MIME"text/plain", choices::ChoiceMap)
    _show_pretty(io, choices, 0, ())
end

# assignments that have static address schemas should also support faster
# accessors, which make the address explicit in the type (Val(:foo) instaed of
# :foo)
function static_get_value end
function static_get_submap end

function _fill_array! end
function _from_array end

"""
    arr::Vector{T} = to_array(choices::ChoiceMap, ::Type{T}) where {T}

Populate an array with values of choices in the given assignment.

It is an error if each of the values cannot be coerced into a value of the
given type.

# Implementation

To support `to_array`, a concrete subtype `T <: ChoiceMap` should implement
the following method:

    n::Int = _fill_array!(choices::T, arr::Vector{V}, start_idx::Int) where {V}

Populate `arr` with values from the given assignment, starting at `start_idx`,
and return the number of elements in `arr` that were populated.
"""
function to_array(choices::ChoiceMap, ::Type{T}) where {T}
    arr = Vector{T}(undef, 32)
    n = _fill_array!(choices, arr, 1)
    @assert n <= length(arr)
    resize!(arr, n)
    arr
end

function _fill_array!(value::T, arr::Vector{T}, start_idx::Int) where {T}
    if length(arr) < start_idx
        resize!(arr, 2 * start_idx)
    end
    arr[start_idx] = value
    1
end

function _fill_array!(value::Vector{T}, arr::Vector{T}, start_idx::Int) where {T}
    if length(arr) < start_idx + length(value)
        resize!(arr, 2 * (start_idx + length(value)))
    end
    arr[start_idx:start_idx+length(value)-1] = value
    length(value)
end


"""
    choices::ChoiceMap = from_array(proto_choices::ChoiceMap, arr::Vector)

Return an assignment with the same address structure as a prototype
assignment, but with values read off from the given array.

The order in which addresses are populated is determined by the prototype
assignment. It is an error if the number of choices in the prototype assignment
is not equal to the length the array.

# Implementation

To support `from_array`, a concrete subtype `T <: ChoiceMap` should implement
the following method:


    (n::Int, choices::T) = _from_array(proto_choices::T, arr::Vector{V}, start_idx::Int) where {V}

Return an assignment with the same address structure as a prototype assignment,
but with values read off from `arr`, starting at position `start_idx`, and the
number of elements read from `arr`.
"""
function from_array(proto_choices::ChoiceMap, arr::Vector)
    (n, choices) = _from_array(proto_choices, arr, 1)
    if n != length(arr)
        error("Dimension mismatch: $n, $(length(arr))")
    end
    choices
end

function _from_array(::T, arr::Vector{T}, start_idx::Int) where {T}
    (1, arr[start_idx])
end

function _from_array(value::Vector{T}, arr::Vector{T}, start_idx::Int) where {T}
    n_read = length(value)
    (n_read, arr[start_idx:start_idx+n_read-1])
end


"""
    choices = Base.merge(choices1::ChoiceMap, choices2::ChoiceMap)

Merge two choice maps.

It is an error if the choice maps both have values at the same address, or if
one choice map has a value at an address that is the prefix of the address of a
value in the other choice map.
"""
function Base.merge(choices1::ChoiceMap, choices2::ChoiceMap)
    choices = DynamicChoiceMap()
    for (key, value) in get_values_shallow(choices1)
        choices.leaf_nodes[key] = value
    end
    for (key, node1) in get_submaps_shallow(choices1)
        node2 = get_submap(choices2, key)
        node = merge(node1, node2)
        choices.internal_nodes[key] = node
    end
    for (key, value) in get_values_shallow(choices2)
        if haskey(choices.leaf_nodes, key)
            error("choices1 has leaf node at $key and choices2 has leaf node at $key")
        end
        if haskey(choices.internal_nodes, key)
            error("choices1 has internal node at $key and choices2 has leaf node at $key")
        end
        choices.leaf_nodes[key] = value
    end
    for (key, node) in get_submaps_shallow(choices2)
        if haskey(choices.leaf_nodes, key)
            error("choices1 has leaf node at $key and choices2 has internal node at $key")
        end
        if !haskey(choices.internal_nodes, key)
            # otherwise it should already be included
            choices.internal_nodes[key] = node
        end
    end
    return choices
end

"""
Variadic merge of choice maps.
"""
function Base.merge(choices1::ChoiceMap, choices_rest::ChoiceMap...)
    reduce(Base.merge, choices_rest; init=choices1)
end

function Base.:(==)(a::ChoiceMap, b::ChoiceMap)
    for (addr, value) in get_values_shallow(a)
        if !has_value(b, addr) || (get_value(b, addr) != value)
            return false
        end
    end
    for (addr, value) in get_values_shallow(b)
        if !has_value(a, addr) || (get_value(a, addr) != value)
            return false
        end
    end
    for (addr, submap) in get_submaps_shallow(a)
        if submap != get_submap(b, addr)
            return false
        end
    end
    for (addr, submap) in get_submaps_shallow(b)
        if submap != get_submap(a, addr)
            return false
        end
    end
    return true
end

function Base.isapprox(a::ChoiceMap, b::ChoiceMap)
    for (addr, value) in get_values_shallow(a)
        if !has_value(b, addr) || !isapprox(get_value(b, addr), value)
            return false
        end
    end
    for (addr, value) in get_values_shallow(b)
        if !has_value(a, addr) || !isapprox(get_value(a, addr), value)
            return false
        end
    end
    for (addr, submap) in get_submaps_shallow(a)
        if !isapprox(submap, get_submap(b, addr))
            return false
        end
    end
    for (addr, submap) in get_submaps_shallow(b)
        if !isapprox(submap, get_submap(a, addr))
            return false
        end
    end
    return true
end


export ChoiceMap
export get_address_schema
export get_submap
export get_value
export has_value
export get_submaps_shallow
export get_values_shallow
export static_get_value
export static_get_submap
export to_array, from_array


######################
# static assignment #
######################

struct StaticChoiceMap{R,S,T,U} <: ChoiceMap
    leaf_nodes::NamedTuple{R,S}
    internal_nodes::NamedTuple{T,U}
    isempty::Bool
end

function StaticChoiceMap{R,S,T,U}(leaf_nodes::NamedTuple{R,S}, internal_nodes::NamedTuple{T,U}) where {R,S,T,U}
    is_empty = length(leaf_nodes) == 0 && all(isempty(n) for n in internal_nodes)
    StaticChoiceMap(leaf_nodes, internal_nodes, is_empty)
end

function StaticChoiceMap(leaf_nodes::NamedTuple{R,S}, internal_nodes::NamedTuple{T,U}) where {R,S,T,U}
    is_empty = length(leaf_nodes) == 0 && all(isempty(n) for n in internal_nodes)
    StaticChoiceMap(leaf_nodes, internal_nodes, is_empty)
end


# invariant: all internal_nodes are nonempty

function get_address_schema(::Type{StaticChoiceMap{R,S,T,U}}) where {R,S,T,U}
    keys = Set{Symbol}()
    for (key, _) in zip(R, S.parameters)
        push!(keys, key)
    end
    for (key, _) in zip(T, U.parameters)
        push!(keys, key)
    end
    StaticAddressSchema(keys)
end

function Base.isempty(choices::StaticChoiceMap)
    choices.isempty
end

get_values_shallow(choices::StaticChoiceMap) = pairs(choices.leaf_nodes)
get_submaps_shallow(choices::StaticChoiceMap) = pairs(choices.internal_nodes)
has_value(choices::StaticChoiceMap, addr::Pair) = _has_value(choices, addr)
get_value(choices::StaticChoiceMap, addr::Pair) = _get_value(choices, addr)
get_submap(choices::StaticChoiceMap, addr::Pair) = _get_submap(choices, addr)

# NOTE: there is no static_has_value because this is known from the static
# address schema

## has_value ##

function has_value(choices::StaticChoiceMap, key::Symbol)
    haskey(choices.leaf_nodes, key)
end

## get_submap ##

function get_submap(choices::StaticChoiceMap, key::Symbol)
    if haskey(choices.internal_nodes, key)
        choices.internal_nodes[key]
    elseif haskey(choices.leaf_nodes, key)
        throw(KeyError(key))
    else
        EmptyChoiceMap()
    end
end

function static_get_submap(choices::StaticChoiceMap, ::Val{A}) where {A}
    choices.internal_nodes[A]
end

## get_value ##

function get_value(choices::StaticChoiceMap, key::Symbol)
    choices.leaf_nodes[key]
end

function static_get_value(choices::StaticChoiceMap, ::Val{A}) where {A}
    choices.leaf_nodes[A]
end

# convert from any other schema that has only Val{:foo} addresses
function StaticChoiceMap(other::ChoiceMap)
    leaf_keys_and_nodes = collect(get_values_shallow(other))
    internal_keys_and_nodes = collect(get_submaps_shallow(other))
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
    StaticChoiceMap(
        NamedTuple{leaf_keys}(leaf_nodes),
        NamedTuple{internal_keys}(internal_nodes),
        isempty(other))
end

"""
    choices = pair(choices1::ChoiceMap, choices2::ChoiceMap, key1::Symbol, key2::Symbol)

Return an assignment that contains `choices1` as a sub-assignment under `key1`
and `choices2` as a sub-assignment under `key2`.
"""
function pair(choices1::ChoiceMap, choices2::ChoiceMap, key1::Symbol, key2::Symbol)
    StaticChoiceMap(NamedTuple(), NamedTuple{(key1,key2)}((choices1, choices2)),
        isempty(choices1) && isempty(choices2))
end

"""
    (choices1, choices2) = unpair(choices::ChoiceMap, key1::Symbol, key2::Symbol)

Return the two sub-assignments at `key1` and `key2`, one or both of which may be empty.

It is an error if there are any top-level values, or any non-empty top-level
sub-assignments at keys other than `key1` and `key2`.
"""
function unpair(choices::ChoiceMap, key1::Symbol, key2::Symbol)
    if !isempty(get_values_shallow(choices)) || length(collect(get_submaps_shallow(choices))) > 2
        error("Not a pair")
    end
    a = get_submap(choices, key1)
    b = get_submap(choices, key2)
    (a, b)
end

# TODO make a generated function?
function _fill_array!(choices::StaticChoiceMap, arr::Vector{T}, start_idx::Int) where {T}
    idx = start_idx
    for value in choices.leaf_nodes
        n_written = _fill_array!(value, arr, idx)
        idx += n_written
    end
    for node in choices.internal_nodes
        n_written = _fill_array!(node, arr, idx)
        idx += n_written
    end
    idx - start_idx
end

@generated function _from_array(
        proto_choices::StaticChoiceMap{R,S,T,U}, arr::Vector{V}, start_idx::Int) where {R,S,T,U,V}
    leaf_node_keys = proto_choices.parameters[1]
    leaf_node_types = proto_choices.parameters[2].parameters
    internal_node_keys = proto_choices.parameters[3]
    internal_node_types = proto_choices.parameters[4].parameters

    exprs = [quote idx = start_idx end]
    leaf_node_names = []
    internal_node_names = []

    # leaf nodes
    for key in leaf_node_keys
        value = gensym()
        push!(leaf_node_names, value)
        push!(exprs, quote
            (n_read, $value) = _from_array(proto_choices.leaf_nodes.$key, arr, idx)
            idx += n_read
        end)
    end

    # internal nodes
    for key in internal_node_keys
        node = gensym()
        push!(internal_node_names, node)
        push!(exprs, quote
            (n_read, $node) = _from_array(proto_choices.internal_nodes.$key, arr, idx)
            idx += n_read
        end)
    end

    quote
        $(exprs...)
        leaf_nodes_field = NamedTuple{R,S}(($(leaf_node_names...),))
        internal_nodes_field = NamedTuple{T,U}(($(internal_node_names...),))
        choices = StaticChoiceMap{R,S,T,U}(leaf_nodes_field, internal_nodes_field)
        (idx - start_idx, choices)
    end
end

@generated function Base.merge(choices1::StaticChoiceMap{R,S,T,U},
                               choices2::StaticChoiceMap{W,X,Y,Z}) where {R,S,T,U,W,X,Y,Z}

    # unpack first assignment type parameters
    leaf_node_keys1 = choices1.parameters[1]
    leaf_node_types1 = choices1.parameters[2].parameters
    internal_node_keys1 = choices1.parameters[3]
    internal_node_types1 = choices1.parameters[4].parameters
    keys1 = (leaf_node_keys1..., internal_node_keys1...,)

    # unpack second assignment type parameters
    leaf_node_keys2 = choices2.parameters[1]
    leaf_node_types2 = choices2.parameters[2].parameters
    internal_node_keys2 = choices2.parameters[3]
    internal_node_types2 = choices2.parameters[4].parameters
    keys2 = (leaf_node_keys2..., internal_node_keys2...,)

    # leaf vs leaf collision is an error
    colliding_leaf_leaf_keys = intersect(leaf_node_keys1, leaf_node_keys2)
    if !isempty(colliding_leaf_leaf_keys)
        error("choices1 and choices2 both have leaf nodes at key(s): $colliding_leaf_leaf_keys")
    end

    # leaf vs internal collision is an error
    colliding_leaf_internal_keys = intersect(leaf_node_keys1, internal_node_keys2)
    if !isempty(colliding_leaf_internal_keys)
        error("choices1 has leaf node and choices2 has internal node at key(s): $colliding_leaf_internal_keys")
    end

    # internal vs leaf collision is an error
    colliding_internal_leaf_keys = intersect(internal_node_keys1, leaf_node_keys2)
    if !isempty(colliding_internal_leaf_keys)
        error("choices1 has internal node and choices2 has leaf node at key(s): $colliding_internal_leaf_keys")
    end

    # internal vs internal collision is not an error, recursively call merge
    colliding_internal_internal_keys = (intersect(internal_node_keys1, internal_node_keys2)...,)
    internal_node_keys1_exclusive = (setdiff(internal_node_keys1, internal_node_keys2)...,)
    internal_node_keys2_exclusive = (setdiff(internal_node_keys2, internal_node_keys1)...,)

    # leaf nodes named tuple
    leaf_node_keys = (leaf_node_keys1..., leaf_node_keys2...,)
    leaf_node_types = map(QuoteNode, (leaf_node_types1..., leaf_node_types2...,))
    leaf_node_values = Expr(:tuple,
        [Expr(:(.), :(choices1.leaf_nodes), QuoteNode(key))
            for key in leaf_node_keys1]...,
        [Expr(:(.), :(choices2.leaf_nodes), QuoteNode(key))
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
        [Expr(:(.), :(choices1.internal_nodes), QuoteNode(key))
            for key in internal_node_keys1_exclusive]...,
        [Expr(:(.), :(choices2.internal_nodes), QuoteNode(key))
            for key in internal_node_keys2_exclusive]...,
        [Expr(:call, :merge,
                Expr(:(.), :(choices1.internal_nodes), QuoteNode(key)),
                Expr(:(.), :(choices2.internal_nodes), QuoteNode(key)))
            for key in colliding_internal_internal_keys]...)
    internal_nodes = Expr(:call,
        Expr(:curly, :NamedTuple, QuoteNode(internal_node_keys)),
        internal_node_values)

    # construct assignment from named tuples
    Expr(:call, :StaticChoiceMap, leaf_nodes, internal_nodes)
end

export StaticChoiceMap
export pair, unpair

#######################
# dynamic assignment #
#######################

struct DynamicChoiceMap <: ChoiceMap
    leaf_nodes::Dict{Any,Any}
    internal_nodes::Dict{Any,Any}
    function DynamicChoiceMap(leaf_nodes::Dict{Any,Any}, internal_nodes::Dict{Any,Any})
        new(leaf_nodes, internal_nodes)
    end
end

# invariant: all internal nodes are nonempty

"""
    struct DynamicChoiceMap <: ChoiceMap .. end

A mutable map from arbitrary hierarchical addresses to values.

    choices = DynamicChoiceMap()

Construct an empty map.

    choices = DynamicChoiceMap(tuples...)

Construct a map containing each of the given (addr, value) tuples.
"""
function DynamicChoiceMap()
    DynamicChoiceMap(Dict(), Dict())
end

function DynamicChoiceMap(tuples...)
    choices = DynamicChoiceMap()
    for (addr, value) in tuples
        choices[addr] = value
    end
    choices
end

"""
    choices = DynamicChoiceMap(other::ChoiceMap)

Copy a choice map, returning a mutable choice map.
"""
function DynamicChoiceMap(other::ChoiceMap)
    choices = DynamicChoiceMap()
    for (addr, val) in get_values_shallow(other)
        choices[addr] = val
    end
    for (addr, submap) in get_submaps_shallow(other)
        set_submap!(choices, addr, DynamicChoiceMap(submap))
    end
    choices
end

"""
    choices = choicemap()

Construct an empty mutable choice map.
"""
function choicemap()
    DynamicChoiceMap()
end

"""
    choices = choicemap(tuples...)

Construct a mutable choice map initialized with given address, value tuples.
"""
function choicemap(tuples...)
    DynamicChoiceMap(tuples...)
end

get_address_schema(::Type{DynamicChoiceMap}) = DynamicAddressSchema()

get_values_shallow(choices::DynamicChoiceMap) = choices.leaf_nodes

get_submaps_shallow(choices::DynamicChoiceMap) = choices.internal_nodes

has_value(choices::DynamicChoiceMap, addr::Pair) = _has_value(choices, addr)

get_value(choices::DynamicChoiceMap, addr::Pair) = _get_value(choices, addr)

get_submap(choices::DynamicChoiceMap, addr::Pair) = _get_submap(choices, addr)

function get_submap(choices::DynamicChoiceMap, addr)
    if haskey(choices.internal_nodes, addr)
        choices.internal_nodes[addr]
    elseif haskey(choices.leaf_nodes, addr)
        throw(KeyError(addr))
    else
        EmptyChoiceMap()
    end
end

has_value(choices::DynamicChoiceMap, addr) = haskey(choices.leaf_nodes, addr)

get_value(choices::DynamicChoiceMap, addr) = choices.leaf_nodes[addr]

function Base.isempty(choices::DynamicChoiceMap)
    isempty(choices.leaf_nodes) && isempty(choices.internal_nodes)
end

# mutation (not part of the assignment interface)

"""
    set_value!(choices::DynamicChoiceMap, addr, value)

Set the given value for the given address.

Will cause any previous value or sub-assignment at this address to be deleted.
It is an error if there is already a value present at some prefix of the given address.

The following syntactic sugar is provided:

    choices[addr] = value
"""
function set_value!(choices::DynamicChoiceMap, addr, value)
    delete!(choices.internal_nodes, addr)
    choices.leaf_nodes[addr] = value
end

function set_value!(choices::DynamicChoiceMap, addr::Pair, value)
    (first, rest) = addr
    if haskey(choices.leaf_nodes, first)
        # we are not writing to the address directly, so we error instead of
        # delete the existing node.
        error("Tried to create assignment at $first but there was already a value there.")
    end
    if haskey(choices.internal_nodes, first)
        node = choices.internal_nodes[first]
    else
        node = DynamicChoiceMap()
        choices.internal_nodes[first] = node
    end
    node = choices.internal_nodes[first]
    set_value!(node, rest, value)
end

"""
    set_submap!(choices::DynamicChoiceMap, addr, submap::ChoiceMap)

Replace the sub-assignment rooted at the given address with the given sub-assignment.
Set the given value for the given address.

Will cause any previous value or sub-assignment at the given address to be deleted.
It is an error if there is already a value present at some prefix of address.
"""
function set_submap!(choices::DynamicChoiceMap, addr, new_node)
    delete!(choices.leaf_nodes, addr)
    delete!(choices.internal_nodes, addr)
    if !isempty(new_node)
        choices.internal_nodes[addr] = new_node
    end
end

function set_submap!(choices::DynamicChoiceMap, addr::Pair, new_node)
    (first, rest) = addr
    if haskey(choices.leaf_nodes, first)
        # we are not writing to the address directly, so we error instead of
        # delete the existing node.
        error("Tried to create assignment at $first but there was already a value there.")
    end
    if haskey(choices.internal_nodes, first)
        node = choices.internal_nodes[first]
    else
        node = DynamicChoiceMap()
        choices.internal_nodes[first] = node
    end
    set_submap!(node, rest, new_node)
end

Base.setindex!(choices::DynamicChoiceMap, value, addr) = set_value!(choices, addr, value)

function _fill_array!(choices::DynamicChoiceMap, arr::Vector{T}, start_idx::Int) where {T}
    leaf_keys_sorted = sort(collect(keys(choices.leaf_nodes)))
    internal_node_keys_sorted = sort(collect(keys(choices.internal_nodes)))
    idx = start_idx
    for key in leaf_keys_sorted
        value = choices.leaf_nodes[key]
        n_written = _fill_array!(value, arr, idx)
        idx += n_written
    end
    for key in internal_node_keys_sorted
        n_written = _fill_array!(get_submap(choices, key), arr, idx)
        idx += n_written
    end
    idx - start_idx
end

function _from_array(proto_choices::DynamicChoiceMap, arr::Vector{T}, start_idx::Int) where {T}
    @assert length(arr) >= start_idx
    choices = DynamicChoiceMap()
    leaf_keys_sorted = sort(collect(keys(proto_choices.leaf_nodes)))
    internal_node_keys_sorted = sort(collect(keys(proto_choices.internal_nodes)))
    idx = start_idx
    for key in leaf_keys_sorted
        (n_read, value) = _from_array(proto_choices.leaf_nodes[key], arr, idx)
        idx += n_read
        choices.leaf_nodes[key] = value
    end
    for key in internal_node_keys_sorted
        (n_read, node) = _from_array(get_submap(proto_choices, key), arr, idx)
        idx += n_read
        choices.internal_nodes[key] = node
    end
    (idx - start_idx, choices)
end

export DynamicChoiceMap
export choicemap
export set_value!
export set_submap!


#######################################
## vector combinator for assignments #
#######################################

# TODO implement LeafVectorChoiceMap, which stores a vector of leaf nodes

struct InternalVectorChoiceMap{T} <: ChoiceMap
    internal_nodes::Vector{T}
    is_empty::Bool
end

function vectorize_internal(nodes::Vector{T}) where {T}
    is_empty = all(map(isempty, nodes))
    InternalVectorChoiceMap(nodes, is_empty)
end

# note some internal nodes may be empty

get_address_schema(::Type{InternalVectorChoiceMap}) = VectorAddressSchema()

Base.isempty(choices::InternalVectorChoiceMap) = choices.is_empty
has_value(choices::InternalVectorChoiceMap, addr::Pair) = _has_value(choices, addr)
get_value(choices::InternalVectorChoiceMap, addr::Pair) = _get_value(choices, addr)
get_submap(choices::InternalVectorChoiceMap, addr::Pair) = _get_submap(choices, addr)

function get_submap(choices::InternalVectorChoiceMap, addr::Int)
    if addr > 0 && addr <= length(choices.internal_nodes)
        choices.internal_nodes[addr]
    else
        EmptyChoiceMap()
    end
end

function get_submaps_shallow(choices::InternalVectorChoiceMap)
    ((i, choices.internal_nodes[i])
     for i=1:length(choices.internal_nodes)
     if !isempty(choices.internal_nodes[i]))
end

get_values_shallow(::InternalVectorChoiceMap) = ()

function _fill_array!(choices::InternalVectorChoiceMap, arr::Vector{T}, start_idx::Int) where {T}
    idx = start_idx
    for key=1:length(choices.internal_nodes)
        n = _fill_array!(choices.internal_nodes[key], arr, idx)
        idx += n
    end
    idx - start_idx
end

function _from_array(proto_choices::InternalVectorChoiceMap{U}, arr::Vector{T}, start_idx::Int) where {T,U}
    @assert length(arr) >= start_idx
    nodes = Vector{U}(undef, length(proto_choices.internal_nodes))
    idx = start_idx
    for key=1:length(proto_choices.internal_nodes)
        (n_read, nodes[key]) = _from_array(proto_choices.internal_nodes[key], arr, idx)
        idx += n_read
    end
    choices = InternalVectorChoiceMap(nodes, proto_choices.is_empty)
    (idx - start_idx, choices)
end

export InternalVectorChoiceMap
export vectorize_internal


####################
# empty assignment #
####################

struct EmptyChoiceMap <: ChoiceMap end

Base.isempty(::EmptyChoiceMap) = true
get_address_schema(::Type{EmptyChoiceMap}) = EmptyAddressSchema()
get_submaps_shallow(::EmptyChoiceMap) = ()
get_values_shallow(::EmptyChoiceMap) = ()

_fill_array!(::EmptyChoiceMap, arr::Vector, start_idx::Int) = 0
_from_array(::EmptyChoiceMap, arr::Vector, start_idx::Int) = (0, EmptyChoiceMap())

export EmptyChoiceMap

############################################
# Nested-dict–like accessor for choicemaps #
############################################

"""
Wrapper for a `ChoiceMap` that provides nested-dict–like syntax, rather than
the default syntax which looks like a flat dict of full keypaths.

```jldoctest
julia> using Gen
julia> c = choicemap((:a, 1),
                     (:b => :c, 2));
julia> cv = nested_view(c);
julia> c[:a] == cv[:a]
true
julia> c[:b => :c] == cv[:b][:c]
true
julia> length(cv)
2
julia> length(cv[:b])
1
julia> sort(collect(keys(cv)))
[:a, :b]
julia> sort(collect(keys(cv[:b])))
[:c]
```
"""
struct ChoiceMapNestedView
    choice_map::ChoiceMap
end

function Base.getindex(choices::ChoiceMapNestedView, addr)
    if has_value(choices.choice_map, addr)
        return get_value(choices.choice_map, addr)
    end
    submap = get_submap(choices.choice_map, addr)
    if isempty(submap)
        throw(KeyError(addr))
    end
    ChoiceMapNestedView(submap)
end

function Base.iterate(c::ChoiceMapNestedView)
    inner_iterator = Base.Iterators.flatten((
        get_values_shallow(c.choice_map),
        ((k, ChoiceMapNestedView(v))
         for (k, v) in get_submaps_shallow(c.choice_map))))
    r = Base.iterate(inner_iterator)
    if r == nothing
        return nothing
    end
    (next_kv, next_inner_state) = r
    (next_kv, (inner_iterator, next_inner_state))
end

function Base.iterate(c::ChoiceMapNestedView, state)
    (inner_iterator, inner_state) = state
    r = Base.iterate(inner_iterator, inner_state)
    if r == nothing
        return nothing
    end
    (next_kv, next_inner_state) = r
    (next_kv, (inner_iterator, next_inner_state))
end

# TODO: Allow different implementations of this method depending on the
# concrete type of the `ChoiceMap`, so that an already-existing data structure
# with faster key lookup (analogous to `Base.KeySet`) can be exposed if it
# exists.
Base.keys(cv::Gen.ChoiceMapNestedView) = (k for (k, v) in cv)

function Base.:(==)(a::ChoiceMapNestedView, b::ChoiceMapNestedView)
  a.choice_map == b.choice_map
end

# Length of a `ChoiceMapNestedView` is number of leaf values + number of
# submaps.  Motivation: This matches what `length` would return for the
# equivalent nested dict.
function Base.length(cv::ChoiceMapNestedView)
  +(get_values_shallow(cv.choice_map) |> collect |> length,
    get_submaps_shallow(cv.choice_map) |> collect |> length)
end

function Base.show(io::IO, ::MIME"text/plain", c::ChoiceMapNestedView)
    Base.show(io, MIME"text/plain"(), c.choice_map)
end

nested_view(c::ChoiceMap) = ChoiceMapNestedView(c)

# TODO(https://github.com/probcomp/Gen/issues/167): Also allow calling
# `nested_view(::Trace)`, to get a nested-dict–like view of the choicemap and
# aux data together.

export nested_view

"""
    selected_choices = get_selected(choices::ChoiceMap, selection::Selection)

Filter the choice map to include only choices in the given selection.

Returns a new choice map.
"""
function get_selected(
        choices::ChoiceMap, selection::Selection)
    output = choicemap()
    for (key, value) in get_values_shallow(choices)
        if (key in selection)
            output[key] = value
        end
    end
    for (key, submap) in get_submaps_shallow(choices)
        subselection = selection[key]
        set_submap!(output, key, get_selected(submap, subselection))
    end
    output
end

export get_selected
