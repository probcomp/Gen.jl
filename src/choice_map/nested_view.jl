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

ChoiceMapNestedView(cm::ValueChoiceMap) = get_value(cm)
ChoiceMapNestedView(::EmptyChoiceMap) = error("Can't convert an emptychoicemap to nested view.")

function Base.getindex(choices::ChoiceMapNestedView, addr)
    ChoiceMapNestedView(get_submap(choices.choice_map, addr))
end

function Base.iterate(c::ChoiceMapNestedView)
    itr = ((k, ChoiceMapNestedView(s)) for (k, s) in get_submaps_shallow(c.choice_map))
    r = Base.iterate(itr)
    if r === nothing
        return nothing
    end
    (next_kv, next_inner_state) = r
    (next_kv, (itr, next_inner_state))
end

function Base.iterate(c::ChoiceMapNestedView, state)
    (itr, st) = state
    r = Base.iterate(itr, st)
    if r === nothing
        return nothing
    end
    (next_kv, next_inner_state) = r
    (next_kv, (itr, next_inner_state))
end

# TODO: Allow different implementations of this method depending on the
# concrete type of the `ChoiceMap`, so that an already-existing data structure
# with faster key lookup (analogous to `Base.KeySet`) can be exposed if it
# exists.
Base.keys(cv::ChoiceMapNestedView) = (k for (k, v) in cv)

Base.:(==)(a::ChoiceMapNestedView, b::ChoiceMapNestedView) = a.choice_map == b.choice_map

function Base.length(cv::ChoiceMapNestedView)
    length(collect(get_submaps_shallow(cv.choice_map)))
end
function Base.show(io::IO, ::MIME"text/plain", c::ChoiceMapNestedView)
    Base.show(io, MIME"text/plain"(), c.choice_map)
end

nested_view(c::ChoiceMap) = ChoiceMapNestedView(c)

# TODO(https://github.com/probcomp/Gen/issues/167): Also allow calling
# `nested_view(::Trace)`, to get a nested-dict–like view of the choicemap and
# aux data together.

export nested_view