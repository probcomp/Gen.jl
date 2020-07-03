"""
    ChoiceMapGetValueError

The error returned when a user attempts to call `get_value`
on an choicemap for an address which does not contain a value in that choicemap.
"""
struct ChoiceMapGetValueError <: Exception end
showerror(io::IO, ex::ChoiceMapGetValueError) = (print(io, "ChoiceMapGetValueError: no value was found for the `get_value` call."))

"""
    ChoiceMap

Abstract type for maps from hierarchical addresses to values.
"""
const ChoiceMap = AddressTree{<:Value}

"""
    get_submaps_shallow(choices::ChoiceMap)

Returns an iterable collection of tuples `(address, submap)`
for each top-level address associated with `choices`.
(This includes `ValueChoiceMap`s.)
"""
@inline get_submaps_shallow(c::ChoiceMap) = get_subtrees_shallow(c)

"""
    get_submap(choices::ChoiceMap, addr)

Return the submap at the given address, or `EmptyChoiceMap`
if there is no submap at the given address.
"""
@inline get_submap(c::ChoiceMap, addr) = get_subtree(c, addr)

"""
    has_value(choices::ChoiceMap)

Returns true if `choices` is a `Value`.

    has_value(choices::ChoiceMap, addr)

Returns true if `choices` has a value stored at address `addr`.
"""
function has_value end
@inline has_value(::ChoiceMap) = false
@inline has_value(c::ChoiceMap, addr) = has_value(get_submap(c, addr))

"""
    get_value(choices::ChoiceMap)

Returns the value stored on `choices` is `choices` is a `ValueChoiceMap`;
throws a `ChoiceMapGetValueError` if `choices` is not a `ValueChoiceMap`.

    get_value(choices::ChoiceMap, addr)
Returns the value stored in the submap with address `addr` or throws
a `ChoiceMapGetValueError` if no value exists at this address.

A syntactic sugar is `Base.getindex`:
    
    value = choices[addr]
"""
function get_value end
@inline get_value(::ChoiceMap) = throw(ChoiceMapGetValueError())
@inline get_value(c::ChoiceMap, addr) = get_value(get_submap(c, addr))
@inline Base.getindex(choices::ChoiceMap, addr...) = get_value(choices, addr...)

"""
    get_values_shallow(choices::ChoiceMap)

Returns an iterable collection of tuples `(address, value)`
for each value stored at a top-level address in `choices`.
(Works by applying a filter to `get_submaps_shallow`,
so this internally requires iterating over every submap.)
"""
function get_values_shallow(choices::ChoiceMap)
    (
        (addr, get_value(submap))
        for (addr, submap) in get_submaps_shallow(choices)
        if has_value(submap)
    )
end

"""
    get_nonvalue_submaps_shallow(choices::ChoiceMap)

Returns an iterable collection of tuples `(address, submap)`
for every top-level submap stored in `choices` which is
not a `ValueChoiceMap`.
(Works by applying a filter to `get_submaps_shallow`,
so this internally requires iterating over every submap.)
"""
function get_nonvalue_submaps_shallow(choices::ChoiceMap)
    (addr_to_submap for addr_to_submap in get_submaps_shallow(choices) if !has_value(addr_to_submap[2]))
end

# support `DynamicChoiceMap` and `StaticChoiceMap` types, and the "legacy" DynamicChoiceMap interface
const DynamicChoiceMap = DynamicAddressTree{Value}
set_submap!(cm::DynamicChoiceMap, addr, submap::ChoiceMap) = set_subtree!(cm, addr, submap)
set_value!(cm::DynamicChoiceMap, addr, val) = set_subtree!(cm, addr, Value(val))
Base.setindex!(cm::DynamicChoiceMap, addr, val) = set_value!(cm, addr, val)

const StaticChoiceMap = StaticAddressTree{Value}

"""
    choices = choicemap()

Construct an empty mutable choice map.
"""
function choicemap()
    DynamicChoiceMap()
end

"""
    choices = choicemap(tuples...)

Construct a mutable choice map initialized with given (address, value) tuples.
(Where `value` is the value to be stored, not a `Value` object.)
"""
function choicemap(tuples...)
    cm = DynamicChoiceMap()
    for (addr, val) in tuples
        set_subtree!(cm, addr, Value(val))
    end
    cm
end

export choicemap
export get_value, has_value
export get_values_shallow, get_submaps_shallow, get_nonvalue_submaps_shallow

include("array_interface.jl")