#######################
# dynamic assignment #
#######################

"""
    struct DynamicChoiceMap AddressTree{Value} .. end

A mutable map from arbitrary hierarchical addresses to values.

    choices = DynamicChoiceMap()

Construct an empty map.

    choices = DynamicChoiceMap(tuples...)

Construct a map containing each of the given (addr, value) tuples.
"""
struct DynamicChoiceMap AddressTree{Value}
    submaps::Dict{Any, ChoiceMap}
    function DynamicChoiceMap()
        new(Dict())
    end
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
    for (addr, submap) in get_submaps_shallow(other)
        if submap isa Value
            set_submap!(choices, addr, submap)
        else
            set_submap!(choices, addr, DynamicChoiceMap(submap))
        end
    end
    choices
end

DynamicChoiceMap(other::Value) = error("Cannot convert a Value to a DynamicChoiceMap")

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

@inline get_submaps_shallow(choices::DynamicChoiceMap) = choices.submaps
@inline get_submap(choices::DynamicChoiceMap, addr) = get(choices.submaps, addr, EmptyChoiceMap())
@inline get_submap(choices::DynamicChoiceMap, addr::Pair) = _get_submap(choices, addr)
@inline Base.isempty(choices::DynamicChoiceMap) = isempty(choices.submaps)

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
    delete!(choices.submaps, addr)
    choices.submaps[addr] = Value(value)
end

function set_value!(choices::DynamicChoiceMap, addr::Pair, value)
    (first, rest) = addr
    if !haskey(choices.submaps, first)
        choices.submaps[first] = DynamicChoiceMap()
    elseif has_value(choices.submaps[first])
        error("Tried to create assignment at $first but there was already a value there.")
    end
    set_value!(choices.submaps[first], rest, value)
end

"""
    set_submap!(choices::DynamicChoiceMap, addr, submap::ChoiceMap)

Replace the sub-assignment rooted at the given address with the given sub-assignment.
Set the given value for the given address.

Will cause any previous value or sub-assignment at the given address to be deleted.
It is an error if there is already a value present at some prefix of address.
"""
function set_submap!(choices::DynamicChoiceMap, addr, new_node::ChoiceMap)
    delete!(choices.submaps, addr)
    if !isempty(new_node)
        choices.submaps[addr] = new_node
    end
end

function set_submap!(choices::DynamicChoiceMap, addr::Pair, new_node::ChoiceMap)
    (first, rest) = addr
    if !haskey(choices.submaps, first)
        choices.submaps[first] = DynamicChoiceMap()
    elseif has_value(choices.submaps[first])
        error("Tried to create assignment at $first but there was already a value there.")
    end
    set_submap!(choices.submaps[first], rest, new_node)
end

Base.setindex!(choices::DynamicChoiceMap, value, addr) = set_value!(choices, addr, value)

function _from_array(proto_choices::DynamicChoiceMap, arr::Vector{T}, start_idx::Int) where {T}
    choices = DynamicChoiceMap()
    keys_sorted = sort(collect(keys(proto_choices.submaps)))
    idx = start_idx
    for key in keys_sorted
        (n_read, submap) = _from_array(proto_choices.submaps[key], arr, idx)
        idx += n_read
        choices.submaps[key] = submap
    end
    (idx - start_idx, choices)
end

get_address_schema(::Type{DynamicChoiceMap}) = DynamicAddressSchema()

export DynamicChoiceMap
export choicemap
export set_value!
export set_submap!