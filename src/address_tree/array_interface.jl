### interface for to_array and fill_array ###

# NOTE: currently this only works for choicemaps,
# but if we found we needed some sort of "to_array" for other types of
# address trees, I don't think it would be too hard to generalize

"""
    arr::Vector{T} = to_array(choices::ChoiceMap, ::Type{T}) where {T}

Populate an array with values of choices in the given assignment.

It is an error if each of the values cannot be coerced into a value of the
given type.

Implementation

The default implmentation of `fill_array` will populate the array by sorting
the addresses of the choicemap using the `sort` function, then iterating over
each submap in this order and filling the array for that submap.

To override the default implementation of `to_array`, 
a concrete subtype  `T <: AddressTree{Value}` should implement the following method:

    n::Int = _fill_array!(choices::T, arr::Vector{V}, start_idx::Int) where {V}

Populate `arr` with values from the given assignment, starting at `start_idx`,
and return the number of elements in `arr` that were populated.

(This is for performance; it is more efficient to fill in values in a preallocated array
by implementing `_fill_array!` than to construct discontiguous arrays for each submap and then merge them.)
"""
function to_array(choices::ChoiceMap, ::Type{T}) where {T}
    arr = Vector{T}(undef, 32)
    n = _fill_array!(choices, arr, 1)
    @assert n <= length(arr)
    resize!(arr, n)
    arr
end

function _fill_array!(c::Value{<:T}, arr::Vector{T}, start_idx::Int) where {T}
    if length(arr) < start_idx
        resize!(arr, 2 * start_idx)
    end
    arr[start_idx] = get_value(c)
    1
end
function _fill_array!(c::Value{<:Vector{<:T}}, arr::Vector{T}, start_idx::Int) where {T}
    value = get_value(c)
    if length(arr) < start_idx + length(value)
        resize!(arr, 2 * (start_idx + length(value)))
    end
    arr[start_idx:start_idx+length(value)-1] = value
    length(value)
end

# default _fill_array! implementation
function _fill_array!(choices::ChoiceMap, arr::Vector{T}, start_idx::Int) where {T}
    key_to_submap = collect(get_submaps_shallow(choices))
    sort!(key_to_submap, by = ((key, submap),) -> key)
    idx = start_idx
    for (key, submap) in key_to_submap
        n_written = _fill_array!(submap, arr, idx)
        idx += n_written
    end
    idx - start_idx
end

"""
    choices::ChoiceMap = from_array(proto_choices::ChoiceMap, arr::Vector)

Return an assignment with the same address structure as a prototype
assignment, but with values read off from the given array.

It is an error if the number of choices in the prototype assignment
is not equal to the length the array.

The order in which addresses are populated with values from the array
should match the order in which the array is populated with values
in a call to `to_array(proto_choices, T)`.  By default,
this means sorting the top-level addresses for `proto_choices`
and then filling in the submaps depth-first in this order.

# Implementation

To support `from_array`, a concrete subtype `T AddressTree{Value}` must implement
the following method:

    (n::Int, choices::T) = _from_array(proto_choices::T, arr::Vector{V}, start_idx::Int) where {V}

Return an assignment with the same address structure as a prototype assignment,
but with values read off from `arr`, starting at position `start_idx`. Return the
number of elements read from `arr`.
"""
function from_array(proto_choices::ChoiceMap, arr::Vector)
    (n, choices) = _from_array(proto_choices, arr, 1)
    if n != length(arr)
        error("Dimension mismatch: $n, $(length(arr))")
    end
    choices
end

function _from_array(::Value, arr::Vector, start_idx::Int)
    (1, Value(arr[start_idx]))
end
function _from_array(c::Value{<:Vector{<:T}}, arr::Vector{T}, start_idx::Int) where {T}
    n_read = length(get_value(c))
    (n_read, Value(arr[start_idx:start_idx+n_read-1]))
end

export to_array, from_array