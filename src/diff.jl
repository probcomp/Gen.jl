###############
# differences #
###############

# NOTE: values encountered during an update that are not wrapped in Diffed are
# assumed to have not changed (i.e. constants)

# NOTE: the requirement is that if a method takes any Diffed arguments, it must 
# return a Diffed value.

# a Julia method may take non-diffed arguments (which indicate no change, in
# the update context) and return a non-diffed arguments. this means that
# constant expressions (and expressions that depend only on contsant
# expressions) can use code that is not overloaded for Diff types.

# Q: what about regular functions that operate on Diffed values. for example,
# suppose there was a method isdiffed(diff::Diffed) that returned true or
# false. can this function not be used in the model?

# A: a different Julia type (e.g. DiffedValue) would need to be used to
# represent Diffed values that do not represent the semantics of Diff. then
# different Julia methods would be implemented that take in Diffed and
# DiffedValues.

# a value with type Diffed should never appear outside of the update context.

"""
    abstract type Diff end

Abstract type for information about a change to a value.
"""
abstract type Diff end

struct UnknownChange <: Diff end 

struct NoChange <: Diff end

struct SetDiff{V} <: Diff

    # elements that were added
    added::Set{V}

    # elements that were deleted
    deleted::Set{V}
end

struct DictDiff{K,V} <: Diff

    # keys that that were added and their values
    added::AbstractDict{K,V}
    
    # keys that were deleted
    deleted::AbstractSet{K}

    # map from key to diff value for that key
    updated::AbstractDict{K,Diff} 
end

struct VectorDiff <: Diff
    new_length::Int
    prev_length::Int
    updated::Dict{Int,Diff} 
end

struct IntDiff <: Diff
    difference::Int # new - old
end

export Diff
export get_diff, strip_diff
export UnknownChange, NoChange
export SetDiff, DictDiff, VectorDiff
export IntDiff


###############################
## differencing of Julia code #
###############################

"""
   Diffed{V,DV <: Diff}

Container for a value and information about a change to its value.
"""
struct Diffed{V,DV <: Diff}
   value::V
   diff::DV
end

# obtain the diff part of a Diffed value
get_diff(diffed::Diffed) = diffed.diff

# a value that is not wrapped in Diffed is a constant
get_diff(value) = NoChange()

strip_diff(diffed::Diffed) = diffed.value

strip_diff(value) = value

export Diffed, diff, strip_diff

# sets

function Base.in(element::Diffed{V,DV}, set::Diffed{T,DT}) where {V, T <: AbstractSet{V}, DV, DT}
    result = strip_diff(element) in strip_diff(set)
    Diffed(result, UnknownChange())
end

function Base.in(element::Diffed{V,NoChange}, set::Diffed{T,NoChange}) where {V, T <: AbstractSet{V}}
    result = strip_diff(element) in strip_diff(set)
    Diffed(result, NoChange())
end

function Base.in(element::Diffed{V,NoChange}, set::Diffed{T, SetDiff{V}}) where {V, T <: AbstractSet{V}}
    el = strip_diff(element)
    result = el in strip_diff(set)
    changed = (el in get_diff(set).added) || (el in get_diff(set).deleted)
    if changed
        Diffed(result, UnknownChange())
    else
        Diffed(result, NoChange())
    end
end

# TODO handle case where the set is itself a constant but the element is a Diffed

function Base.length(set::Diffed{T,UnknownChange}) where {V, T <: AbstractSet{V}}
    result = length(strip_diff(set))
    Diffed(result, UnknownChange())
end

function Base.length(set::Diffed{T,NoChange}) where {V, T <: AbstractSet{V}}
    result = length(strip_diff(set))
    Diffed(result, NoChange())
end

function Base.length(set::Diffed{T,SetDiff{V}}) where {V, T <: AbstractSet{V}}
    result = length(strip_diff(set))
    n_added = length(get_diff(set).added)
    n_deleted = length(get_diff(set).deleted)
    if n_added != n_deleted
        Diffed(result, IntDiff(n_added - n_deleted))
    else
        Diffed(result, NoChange())
    end
end

# dictionaries

function Base.haskey(dict::Diffed{T,DT}, key::Diffed{K,DK}) where {K, V, T <: AbstractDict{K,V}, DT, DK}
    result = haskey(strip_diff(dict), strip_diff(key))
    Diffed(result, UnknownChange())
end

function Base.haskey(dict::Diffed{T,NoChange}, key::Union{Diffed{K,NoChange},K}) where {K, V, T <: AbstractDict{K,V}}
    result = haskey(strip_diff(dict), strip_diff(key))
    Diffed(result, NoChange())
end

function Base.haskey(dict::Diffed{T,DictDiff{K,V}}, key::Union{Diffed{K,NoChange},K}) where {K, V, T <: AbstractDict{K,V}}
    result = haskey(strip_diff(dict), key)
    changed = (key in get_diff(dict).deleted) || haskey(get_diff(dict).added, key)
    if changed
        Diffed(result, UnknownChange())
    else
        Diffed(has, NoChange())
    end
end

function Base.getindex(dict::Diffed{T,DT}, key::Diffed{K,DK}) where {K, V, T  <: AbstractDict{K,V}, DT, DK}
    result = getindex(strip_diff(dict), key)
    Diffed(result, UnknownChange())
end

function Base.getindex(dict::Diffed{T,NoChange}, key::Union{Diffed{K,NoChange},K}) where {K, V, T <: AbstractDict{K,V}}
    result = getindex(strip_diff(dict), key)
    Diffed(result, NoChange())
end

function Base.getindex(dict::Diffed{T,DictDiff{K,V}}, key::Union{Diffed{K,NoChange},K}) where {K, V, T <: AbstractDict{K,V}}
    result = getindex(strip_diff(dict), key)
    changed = (key in get_diff(dict).deleted) || haskey(get_diff(dict).added, key)
    if changed
        Diffed(result, UnknownChange())
    else
        Diffed(val, NoChange())
    end
end

# TODO handle case where the dictionary is itself a constant, but the key is a Diffed

# vectors and tuples

function Base.length(vec::Diffed{T,UnknownChange}) where {T <: Union{AbstractVector,Tuple}}
    result = length(strip_diff(vec))
    Diffed(result, UnknownChange())
end

function Base.length(vec::Diffed{T,NoChange}) where {T <: Union{AbstractVector,Tuple}}
    result = length(strip_diff(vec))
    Diffed(result, NoChange())
end

function Base.length(vec::Diffed{T,VectorDiff}) where {T <: Union{AbstractVector,Tuple}}
    len = length(strip_diff(vec))
    len_diff = get_diff(vec).new_length - get_diff(vec).prev_length
    if len_diff == 0
        Diffed(len, NoChange())
    else
        Diffed(len, IntDiff(len_diff))
    end
end

# TODO: we know that indeices before teh deleted/inserted idnex have not been changed

function Base.getindex(vec::Union{AbstractVector,Tuple}, idx::Diffed{U,DU}) where {U <: Integer, DU}
    result = vec[strip_diff(idx)]
    Diffed(result, UnknownChange())
end

function Base.getindex(vec::Union{AbstractVector,Tuple}, idx::Diffed{U,NoChange}) where {U <: Integer}
    result = vec[strip_diff(idx)]
    Diffed(result, NoChange())
end

function Base.getindex(vec::Diffed{T,DT}, idx::Union{Diffed{U,DU},Integer}) where {T <: Union{AbstractVector,Tuple}, U <: Integer, DT, DU}
    result = strip_diff(vec)[strip_diff(idx)]
    Diffed(result, UnknownChange())
end

function Base.getindex(vec::Diffed{T,NoChange}, idx::Union{Diffed{U,NoChange},Integer}) where {T <: Union{AbstractVector,Tuple}, U <: Integer}
    result = strip_diff(vec)[strip_diff(idx)]
    Diffed(result, NoChange())
end

function Base.getindex(vec::Diffed{T,VectorDiff}, idx::Union{Diffed{U,NoChange},Integer}) where {T <: Union{AbstractVector,Tuple}, U <: Integer}
    v = strip_diff(vec)
    i = strip_diff(idx)
    d = get_diff(vec)
    result = v[i]
    if i > d.prev_length
        Diffed(result, UnknownChange()) 
    elseif haskey(d.updated, i)
        Diffed(result, d.updated[i])
    else
        Diffed(result, NoChange())
    end
end

# TODO handle case where the vector or tuple is itself a constant, but the index is a Diffed

# binary operators 

macro diffed_binary_operator(op)
    quote
        function $(op)(a::Diffed{T,NoChange}, b::Diffed{U,NoChange}) where {T,U}
            result = $(op)(strip_diff(a), strip_diff(b))
            Diffed(result, NoChange())
        end

        function $(op)(a, b::Diffed{U,NoChange}) where {U}
            result = $(op)(a, strip_diff(b))
            Diffed(result, NoChange())
        end

        function $(op)(a::Diffed{T,NoChange}, b) where {T}
            result = $(op)(strip_diff(a), b)
            Diffed(result, NoChange())
        end

        function $(op)(a::Diffed{T,DT}, b::Diffed{U,DU}) where {T,U,DT,DU}
            result = $(op)(strip_diff(a), strip_diff(b))
            Diffed(result, UnknownChange())
        end

        function $(op)(a, b::Diffed{U,UnknownChange}) where {U}
            result = $(op)(a, strip_diff(b))
            Diffed(result, UnknownChange())
        end

        function $(op)(a::Diffed{T,UnknownChange}, b) where {T}
            result = $(op)(strip_diff(a), b)
            Diffed(result, UnknownChange())
        end

        function $(op)(a::Diffed{T,UnknownChange}, b::Diffed{U,UnknownChange}) where {T,U}
            result = $(op)(strip_diff(a), strip_diff(b))
            Diffed(result, UnknownChange())
        end
    end
end

@diffed_binary_operator Base.:+
@diffed_binary_operator Base.:-
@diffed_binary_operator Base.:/

# TODO use a macro to generate this code for +, *, /, -, ==, 

# fill

function Base.fill(value::Diffed{V,NoChange}, n::Union{Diffed{U,NoChange},Integer}) where {V,U <: Integer}
    result = fill(strip_diff(value), strip_diff(n))
    Diffed(result, NoChange())
end

function Base.fill(value::V, n::Diffed{U,NoChange}) where {V,U <: Integer}
    result = fill(value, strip_diff(n))
    Diffed(result, NoChange())
end

function Base.fill(value::Diffed{V,DV}, n::Diffed{U,DU}) where {V,U <: Integer,DU,DV}
    result = fill(strip_diff(value), strip_diff(n))
    Diffed(result, UnknownChange())
end

# TODO filter, map, reduce, foldl, etc.

# NOTE: just handle the case where the function argument is a constant (a Function not a Diffed{Function,Diff})

# broadcasting (?)
