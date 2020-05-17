# Choice Maps

Maps from the addresses of random choices to their values are stored in associative tree-structured data structures that have the following abstract type:
```@docs
ChoiceMap
```

Choice maps are constructed by users to express observations and/or constraints on the traces of generative functions.
Choice maps are also returned by certain Gen inference methods, and are used internally by various Gen inference methods.

A choicemap a tree, whose leaf nodes store a single value, and whose internal nodes provide addresses
for sub-choicemaps.  Leaf nodes have type:
```@docs
ValueChoiceMap
```

Choice maps provide the following methods:
```@docs
get_submap
get_submaps_shallow
has_value
get_value
get_values_shallow
get_nonvalue_submaps_shallow
to_array
from_array
get_selected
```
Note that none of these methods mutate the choice map.

Choice maps also implement:

- `Base.isempty`, which tests of there are no random choices in the choice map

- `Base.merge`, which takes two choice maps, and returns a new choice map containing all random choices in either choice map. It is an error if the choice maps both have values at the same address, or if one choice map has a value at an address that is the prefix of the address of a value in the other choice map.

- `==`, which tests if two choice maps have the same addresses and values at those addresses.


## Mutable Choice Maps

A mutable choice map can be constructed with [`choicemap`](@ref), and then populated:
```julia
choices = choicemap()
choices[:x] = true
choices["foo"] = 1.25
choices[:y => 1 => :z] = -6.3
```

There is also a constructor that takes initial (address, value) pairs:
```julia
choices = choicemap((:x, true), ("foo", 1.25), (:y => 1 => :z, -6.3))
```

```@docs
choicemap
set_value!
set_submap!
```

## Implementing custom choicemap types

To implement a custom choicemap, one must implement
`get_submap` and `get_submaps_shallow`.
To avoid method ambiguity with the default
`get_submap(::ChoiceMap, ::Pair)`, one must implement both
```julia
get_submap(::CustomChoiceMap, addr)
```
and
```julia
get_submap(::CustomChoiceMap, addr::Pair)
```
To use the default implementation of `get_submap(_, ::Pair)`,
one may define
```julia
get_submap(c::CustomChoiceMap, addr::Pair) = _get_choicemap(c, addr)
```

Once `get_submap` and `get_submaps_shallow` are defined, default 
implementations are provided for:
- `has_value`
- `get_value`
- `get_values_shallow`
- `get_nonvalue_submaps_shallow`
- `to_array`
- `get_selected`

If one wishes to support `from_array`, they must implement 
`_from_array`, as described in the documentation for
[`from_array`](@ref).