###################
# address schemas #
###################

abstract type AddressSchema end

struct StaticAddressSchema <: AddressSchema
    keys::Set{Symbol}
end

Base.keys(schema::StaticAddressSchema) = schema.keys

struct VectorAddressSchema <: AddressSchema end
struct SingleDynamicKeyAddressSchema <: AddressSchema end
struct DynamicAddressSchema <: AddressSchema end
struct EmptyAddressSchema <: AddressSchema end
struct AllAddressSchema <: AddressSchema end

export AddressSchema
export StaticAddressSchema # hierarchical
export VectorAddressSchema # hierarchical
export SingleDynamicKeyAddressSchema # hierarchical
export DynamicAddressSchema # hierarchical
export EmptyAddressSchema
export AllAddressSchema

######################
# abstract selection #
######################

"""
    abstract type Selection end

Abstract type for selections of addresses.

All selections implement the following methods:

    Base.in(addr, selection)

Is the address selected?

    Base.getindex(selection, addr)

Get the subselection at the given address.

    Base.isempty(selection)

Is the selection guaranteed to be empty?

    get_address_schema(T)

Return a shallow, compile-time address schema, where `T` is the concrete type of the selection.
"""
abstract type Selection end

Base.in(addr, ::Selection) = false
Base.getindex(::Selection, addr) = EmptySelection()

export Selection

##########################
# hierarchical selection #
##########################

"""
    abstract type HierarchicalSelection <: Selection end

Abstract type for selections that have a notion of sub-selections.

    get_subselections(selection::HierarchicalSelection)

Return an iterator over pairs of addresses and subselections at associated addresses.
"""
abstract type HierarchicalSelection <: Selection end

export HierarchicalSelection
export get_subselections

###################
# empty selection #
###################

"""
    struct EmptySelection <: Selection end

A singleton type for a selection that is always empty.
"""
struct EmptySelection <: Selection end
get_address_schema(::Type{EmptySelection}) = EmptyAddressSchema()
Base.isempty(::EmptySelection) = true

export EmptySelection

#################
# all selection #
#################

"""
    struct AllSelection <: Selection end

A singleton type for a selection that contains all choices at or under an address.
"""
struct AllSelection <: Selection end
get_address_schema(::Type{AllSelection}) = AllAddressSchema()
Base.isempty(::AllSelection) = false # it is not guaranteed to be empty
Base.in(addr, ::AllSelection) = true
Base.getindex(::AllSelection, addr) = AllSelection()

export AllSelection

########################
# complement selection #
########################

struct ComplementSelection <: Selection
    complement::Selection
end
get_address_schema(::Type{ComplementSelection}) = DynamicAddressSchema()
Base.isempty(::ComplementSelection) = false # it is not guaranteed to be empty
Base.in(addr, selection::ComplementSelection) = !(addr in selection.complement)
function Base.getindex(selection::ComplementSelection, addr)
    ComplementSelection(selection.complement[addr])
end

"""
    comp_selection = complement(selection::Selection)

Return a selection that is the complement of the given selection.

An address is in the selection if it is not in the complement selection.
"""
function complement(selection::Selection)
    ComplementSelection(selection)
end

export ComplementSelection, complement

####################
# static selection #
####################

# R is a tuple of symbols..
# T is a tuple of symbols
# U the tuple type of subselections

"""
    struct StaticSelection{T,U} <: HierarchicalSelection .. end

A hierarchical selection whose keys are among its type parameters.
"""
struct StaticSelection{T,U} <: HierarchicalSelection
    subselections::NamedTuple{T,U}
end

function Base.isempty(selection::StaticSelection{T,U}) where {T,U}
    length(R) == 0 && all(isempty(node) for node in selection.subselections)
end

function get_address_schema(::Type{StaticSelection{T,U}}) where {T,U}
    keys = Set{Symbol}()
    for (key, _) in zip(T, U.parameters)
        push!(keys, key)
    end
    StaticAddressSchema(keys)
end

get_subselections(selection::StaticSelection) = pairs(selection.subselections)

function static_getindex(selection::StaticSelection, ::Val{A}) where {A}
    selection.subselections[A]
end

# TODO do we no longer need static_in?

function Base.getindex(selection::StaticSelection, addr::Symbol)
    if haskey(selection.subselections, addr)
        selection.subselections[addr]
    else
        EmptySelection()
    end
end

function Base.getindex(selection::StaticSelection, addr::Pair)
    (first, rest) = addr
    subselection = selection.subselections[first]
    subselection[rest]
end

function Base.in(addr::Symbol, selection::StaticSelection{T,U}) where {T,U}
    addr in T && selection.subselections[addr] == AllSelection()
end

function Base.in(addr::Pair, selection::StaticSelection)
    (first, rest) = addr
    if haskey(selection.subselections, first)
        subselection = selection.subselections[first]
        in(subselection, rest)
    else
        false
    end
end

function StaticSelection(other::HierarchicalSelection)
    keys_and_subselections = collect(get_subselections(other))
    if length(keys_and_subselections) > 0
        (keys, subselections) = collect(zip(keys_and_subselections...))
    else
        (keys, subselections) = ((), ())
    end
    types = map(typeof, subselections)
    StaticSelection{keys,Tuple{types...}}(NamedTuple{keys}(subselections))
end

export StaticSelection


#####################
# dynamic selection #
#####################

"""
    struct DynamicSelection <: HierarchicalSelection .. end

A hierarchical, mutable, selection with arbitrary addresses.

Can be mutated with the following methods:


    Base.push!(selection::DynamicSelection, addr)

Add the address and all of its sub-addresses to the selection.

Example:
```julia
selection = select()
@assert !(:x in selection)
push!(selection, :x)
@assert :x in selection
```

    set_subselection!(selection::DynamicSelection, addr, other::Selection)

Change the selection status of the given address and its sub-addresses that defined by `other`.

Example:
```julia
selection = select(:x)
@assert :x in selection
subselection = select(:y)
set_subselection!(selection, :x, subselection)
@assert (:x => :y) in selection
@assert !(:x in selection)
```

Note that `set_subselection!` does not copy data in `other`, so `other` may be mutated by a later calls to `set_subselection!` for addresses under `addr`.
"""
struct DynamicSelection <: HierarchicalSelection
    # note: only store subselections for which isempty = false
    subselections::Dict{Any,Selection}
end

function Base.isempty(selection::DynamicSelection)
    isempty(selection.subselections)
end

DynamicSelection() = DynamicSelection(Dict{Any,Selection}())

get_address_schema(::Type{DynamicSelection}) = DynamicAddressSchema()

function Base.in(addr, selection::DynamicSelection)
    if haskey(selection.subselections, addr)
        selection.subselections[addr] == AllSelection()
    else
        false
    end
end

function Base.in(addr::Pair, selection::DynamicSelection)
    (first, rest) = addr
    if haskey(selection.subselections, first)
        subselection = selection.subselections[first]
        @assert !isempty(subselection)
        rest in subselection
    else
        false
    end
end

function Base.getindex(selection::DynamicSelection, addr)
    if haskey(selection.subselections, addr)
        selection.subselections[addr]
    else
        EmptySelection()
    end
end

function Base.getindex(selection::DynamicSelection, addr::Pair)
    (first, rest) = addr
    if haskey(selection.subselections, first)
        subselection = selection.subselections[first]
        @assert !isempty(subselection)
        getindex(subselection, rest)
    else
        EmptySelection()
    end
end

function Base.push!(selection::DynamicSelection, addr)
    selection.subselections[addr] = AllSelection()
end

function Base.push!(selection::DynamicSelection, addr::Pair)
    (first, rest) = addr
    if haskey(selection.subselections, first)
        subselection = selection.subselections[first]
    else
        subselection = DynamicSelection()
        selection.subselections[first] = subselection
    end
    push!(subselection, rest)
end

function set_subselection!(selection::DynamicSelection, addr, other::Selection)
    selection.subselections[addr] = other
end

function set_subselection!(selection::DynamicSelection, addr::Pair, other::Selection)
    (first, rest) = addr
    if haskey(selection.subselections, first)
        subselection = selection.subselections[first]
    else
        subselection = DynamicSelection()
        selection.subselections[first] = subselection
    end
    set_subselection!(subselection, rest, other)
end

get_subselections(selection::DynamicSelection) = selection.subselections

"""
    selection = select(addrs...)

Return a selection containing a given set of addresses.

Examples:
```julia
selection = select(:x, "foo", :y => 1 => :z)
selection = select()
selection = select(:x => 1, :x => 2)
```
"""
function select(addrs...)
    selection = DynamicSelection()
    for addr in addrs
        push!(selection, addr)
    end
    selection
end

"""
    selection = selectall()

Construct a selection that includes all random choices.
"""
function selectall()
    AllSelection()
end

export DynamicSelection
export select, selectall, set_subselection!
