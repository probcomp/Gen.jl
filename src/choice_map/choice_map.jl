#########################
# choice map interface #
#########################

"""
    ChoiceMapGetValueError

The error returned when a user attempts to call `get_value`
on an choicemap for an address which does not contain a value in that choicemap.
"""
struct ChoiceMapGetValueError <: Exception end
showerror(io::IO, ex::ChoiceMapGetValueError) = (print(io, "ChoiceMapGetValueError: no value was found for the `get_value` call."))

"""
    abstract type ChoiceMap end

Abstract type for maps from hierarchical addresses to values.
"""
abstract type ChoiceMap end

"""
    get_submaps_shallow(choices::ChoiceMap)

Returns an iterable collection of tuples `(address, submap)`
for each top-level address associated with `choices`.
(This includes `ValueChoiceMap`s.)
"""
function get_submaps_shallow end

"""
    get_submap(choices::ChoiceMap, addr)

Return the submap at the given address, or `EmptyChoiceMap`
if there is no submap at the given address.
"""
function get_submap end

# provide _get_submap so when users overwrite get_submap(choices::CustomChoiceMap, addr::Pair)
# they can just call _get_submap for convenience if they want
@inline function _get_submap(choices::ChoiceMap, addr::Pair)
    (first, rest) = addr
    submap = get_submap(choices, first)
    get_submap(submap, rest)
end

"""
    has_value(choices::ChoiceMap)

Returns true if `choices` is a `ValueChoiceMap`.

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
    has_submap(choices::ChoiceMap, addr)
Return true if there is a non-empty sub-assignment at the given address.
"""
function has_submap end
@inline has_submap(choices::ChoiceMap, addr) = !isempty(get_submap(choices, addr))

"""
schema = get_address_schema(::Type{T}) where {T <: ChoiceMap}

Return the (top-level) address schema for the given choice map.
"""
function get_address_schema end

# get_values_shallow and get_nonvalue_submaps_shallow are just filters on get_submaps_shallow
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

# a choicemap is empty if it has no submaps and no value
Base.isempty(c::ChoiceMap) = all(((addr, submap),) -> isempty(submap), get_submaps_shallow(c)) && !has_value(c)

"""
    EmptyChoiceMap

A choicemap with no submaps or values.
"""
struct EmptyChoiceMap <: ChoiceMap end

@inline has_value(::EmptyChoiceMap, addr...) = false
@inline get_value(::EmptyChoiceMap) = throw(ChoiceMapGetValueError())
@inline get_submap(::EmptyChoiceMap, addr) = EmptyChoiceMap()
@inline Base.isempty(::EmptyChoiceMap) = true
@inline get_submaps_shallow(::EmptyChoiceMap) = ()
@inline get_address_schema(::Type{EmptyChoiceMap}) = EmptyAddressSchema()
@inline Base.:(==)(::EmptyChoiceMap, ::EmptyChoiceMap) = true
@inline Base.:(==)(::ChoiceMap, ::EmptyChoiceMap) = false
@inline Base.:(==)(::EmptyChoiceMap, ::ChoiceMap) = false

"""
    ValueChoiceMap

A leaf-node choicemap.  Stores a single value.
"""
struct ValueChoiceMap{T} <: ChoiceMap
    val::T
end

@inline has_value(choices::ValueChoiceMap) = true
@inline get_value(choices::ValueChoiceMap) = choices.val
@inline get_submap(choices::ValueChoiceMap, addr) = EmptyChoiceMap()
@inline get_submaps_shallow(choices::ValueChoiceMap) = ()
@inline Base.:(==)(a::ValueChoiceMap, b::ValueChoiceMap) = a.val == b.val
@inline Base.isapprox(a::ValueChoiceMap, b::ValueChoiceMap) = isapprox(a.val, b.val)
@inline get_address_schema(::Type{<:ValueChoiceMap}) = AllAddressSchema()

"""
    choices = Base.merge(choices1::ChoiceMap, choices2::ChoiceMap)

Merge two choice maps.

It is an error if the choice maps both have values at the same address, or if
one choice map has a value at an address that is the prefix of the address of a
value in the other choice map.
"""
function Base.merge(choices1::ChoiceMap, choices2::ChoiceMap)
    choices = DynamicChoiceMap()
    for (key, submap) in get_submaps_shallow(choices1)
        set_submap!(choices, key, merge(submap, get_submap(choices2, key)))
    end
    for (key, submap) in get_submaps_shallow(choices2)
        if isempty(get_submap(choices1, key))
            set_submap!(choices, key, submap)
        end
    end
    choices
end
Base.merge(c::ChoiceMap, ::EmptyChoiceMap) = c
Base.merge(::EmptyChoiceMap, c::ChoiceMap) = c
Base.merge(c::ValueChoiceMap, ::EmptyChoiceMap) = c
Base.merge(::EmptyChoiceMap, c::ValueChoiceMap) = c
Base.merge(::ValueChoiceMap, ::ChoiceMap) = error("ValueChoiceMaps cannot be merged")
Base.merge(::ChoiceMap, ::ValueChoiceMap) = error("ValueChoiceMaps cannot be merged")

"""
Variadic merge of choice maps.
"""
function Base.merge(choices1::ChoiceMap, choices_rest::ChoiceMap...)
    reduce(Base.merge, choices_rest; init=choices1)
end

function Base.:(==)(a::ChoiceMap, b::ChoiceMap)
    for (addr, submap) in get_submaps_shallow(a)
        if get_submap(b, addr) != submap
            return false
        end
    end
    for (addr, submap) in get_submaps_shallow(b)
        if get_submap(a, addr) != submap
            return false
        end
    end
    return true
end

# This is modeled after
# https://github.com/JuliaLang/julia/blob/7bff5cdd0fab8d625e48b3a9bb4e94286f2ba18c/base/abstractdict.jl#L530-L537
const hasha_seed = UInt === UInt64 ? 0x6d35bb51952d5539 : 0x952d5539
function Base.hash(a::ChoiceMap, h::UInt)
    hv = hasha_seed
    for (addr, value) in get_values_shallow(a)
        hv = xor(hv, hash(addr, hash(value)))
    end
    for (addr, submap) in get_submaps_shallow(a)
        hv = xor(hv, hash(addr, hash(submap)))
    end
    return hash(hv, h)
end

function Base.isapprox(a::ChoiceMap, b::ChoiceMap)
    for (addr, submap) in get_submaps_shallow(a)
        if !isapprox(get_submap(b, addr), submap)
            return false
        end
    end
    return true
end

"""
    selected_choices = get_selected(choices::ChoiceMap, selection::Selection)

Filter the choice map to include only choices in the given selection.

Returns a new choice map.
"""
function get_selected(
        choices::ChoiceMap, selection::Selection)
    # TODO: return a `FilteringChoiceMap` which does this filtering lazily!
    output = choicemap()
    for (addr, submap) in get_submaps_shallow(choices)
        if has_value(submap) && addr in selection
            output[addr] = get_value(submap)
        else
            subselection = selection[addr]
            set_submap!(output, addr, get_selected(submap, subselection))
        end
    end
    output
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
    key_and_submaps = collect(get_nonvalue_submaps_shallow(choices))
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

export ChoiceMap, ValueChoiceMap, EmptyChoiceMap
export _get_submap, get_submap, get_submaps_shallow, has_submap
export get_value, has_value
export get_values_shallow, get_nonvalue_submaps_shallow
export get_address_schema, get_selected
export ChoiceMapGetValueError

include("array_interface.jl")
include("dynamic_choice_map.jl")
include("static_choice_map.jl")
include("nested_view.jl")