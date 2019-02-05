# Choice Maps

Maps from the addresses of random choices to their values are stored in associative tree-structured data structures that have the following abstract type:
```@docs
ChoiceMap
```

Choice maps are constructed by users to express observations and/or constraints on the traces of generative functions.
Choice maps are also returned by certain Gen inference methods, and are used internally by various Gen inference methods.

Choice maps provide the following methods:
```@docs
has_value
get_value
get_submap
get_values_shallow
get_submaps_shallow
to_array
from_array
address_set
```
Note that none of these methods mutate the assignment.

Choice maps also provide `Base.isempty`, which tests of there are no random
choices in the assignment, and `Base.merge`, which takes two assignments, and
returns a new assignment containing all random choices in either assignment.
It is an error if the assignments both have values at the same address, or if
one assignment has a value at an address that is the prefix of the address of a
value in the other assignment.


## Dynamic Choice Map 

One concrete assignment type is `DynamicChoiceMap`, which is mutable.
Users construct `DynamicChoiceMaps` and populate them for use as observations or constraints, e.g.:
```julia
choices = choicemap()
choices[:x] = true
choices["foo"] = 1.25
choices[:y => 1 => :z] = -6.3
```

There is also a constructor for `DynamicChoiceMap` that takes initial (address, value) pairs:
```julia
choices = choicemap((:x, true), ("foo", 1.25), (:y => 1 => :z, -6.3))
```

```@docs
choicemap
set_value!
set_submap!
```
