#########################
# choice map interface #
#########################

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
@inline get_submap(choices::ChoiceMap, addr::Pair) = _get_submap(choices, addr)

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
throws a `KeyError` if `choices` is not a `ValueChoiceMap`.

    get_value(choices::ChoiceMap, addr)
Returns the value stored in the submap with address `addr` or throws
a `KeyError` if no value exists at this address.

A syntactic sugar is `Base.getindex`:
    
    value = choices[addr]
"""
function get_value end
get_value(::ChoiceMap) = throw(KeyError(nothing))
get_value(c::ChoiceMap, addr) = get_value(get_submap(c, addr))
@inline Base.getindex(choices::ChoiceMap, addr...) = get_value(choices, addr...)

# get_values_shallow and get_nonvalue_submaps_shallow are just filters on get_submaps_shallow
"""
    get_values_shallow(choices::ChoiceMap)

Returns an iterable collection of tuples `(address, value)`
for each value stored at a top-level address in `choices`.
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
"""
function get_nonvalue_submaps_shallow(choices::ChoiceMap)
    filter(! ∘ has_value, get_submaps_shallow(choices))
end

# a choicemap is empty if it has no submaps and no value
Base.isempty(c::ChoiceMap) = isempty(get_submaps_shallow(c)) && !has_value(c)

"""
    abstract type ChoiceMap end

Abstract type for maps from hierarchical addresses to values.
"""
abstract type ChoiceMap end

"""
    EmptyChoiceMap

A choicemap with no submaps or values.
"""
struct EmptyChoiceMap <: ChoiceMap end

@inline has_value(::EmptyChoiceMap, addr...) = false
@inline get_value(::EmptyChoiceMap) = throw(KeyError(nothing))
@inline get_submap(::EmptyChoiceMap, addr) = EmptyChoiceMap()
@inline Base.isempty(::EmptyChoiceMap) = true
@inline get_submaps_shallow(::EmptyChoiceMap) = ()

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
Base.:(==)(a::ValueChoiceMap, b::ValueChoiceMap) = a.val == b.val
Base.isapprox(a::ValueChoiceMap, b::ValueChoiceMap) = isapprox(a.val, b.val)

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
    return true
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
export get_submap, get_submaps_shallow
export get_value, has_value
export get_values_shallow, get_nonvalue_submaps_shallow

include("array_interface.jl")
include("dynamic_choice_map.jl")
include("static_choice_map.jl")
include("nested_view.jl")