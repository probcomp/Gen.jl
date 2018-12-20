# Reference

## Modeling Languages

### Addresses of Random Choices

### Dynamic DSL

### Static DSL

### Combinators

## Assignments

An *assignment* is a map from addresses of random choices to their values.
Assignments are constructed by users to express observations and/or constraints on the traces of generative functions.
Assignments are also returned by certain Gen inference methods, and are used internally by various Gen inference methods.

There are various concrete types for assignments, each of which is a subtype of `Assignment`.
Assignments provide the following methods:
```@docs
has_value
get_value
get_subassmt
get_values_shallow
get_subassmts_shallow
to_array
from_array
```
Note that none of these methods mutate the assignment.

Assignments also provide `Base.isempty`, which tests of there are no random
choices in the assignment, and `Base.merge`, which takes two assignments, and
returns a new assignment containing all random choices in either assignment.
It is an error if the assignments both have values at the same address, or if
one assignment has a value at an address that is the prefix of the address of a
value in the other assignment.


### Dynamic Assignment

One concrete assignment type is `DynamicAssignment`, which is mutable.
Users construct `DynamicAssignments` and populate them for use as observations or constraints, e.g.:
```julia
assmt = DynamicAssignment()
assmt[:x] = true
assmt["foo"] = 1.25
assmt[:y => 1 => :z] = -6.3
```

```@docs
DynamicAssignment
set_value!
set_subassmt!
```

## Selections

A *selection* is a set of addresses.
Users typically construct selections and pass them to Gen inference library methods.

There are various concrete types for selections, each of which is a subtype of `AddressSet`.
One such concrete type is `DynamicAddressSet`, which users can populate using `Base.push!`, e.g.:
```julia
sel = DynamicAddressSet()
push!(sel, :x)
push!(sel, "foo")
push!(sel, :y => 1 => :z)
```
There is also the following syntactic sugar:
```julia
sel = select(:x, "foo", :y => 1 => :z)
```


## Inference Library

### Importance Sampling
```@docs
importance_sampling
importance_resampling
```

### Markov Chain Monte Carlo
```@docs
default_mh
simple_mh
custom_mh
general_mh
mala
hmc
```

### Optimization over Random Choices
```@docs
map_optimize
```

### Particle Filtering
```@docs
particle_filter_default
particle_filter_custom
```

### Training Generative Functions
```@docs
sgd_train_batch
```


## Generative Function Interface

A *trace* is a record of an execution of a generative function.
There is no abstract type representing all traces.
Generative functions implement the *generative function interface*, which is a set of methods that involve the execution traces and probabilistic behavior of generative functions.
In the mathematical description of the interface methods, we denote arguments to a function by ``x``, complete assignments of values to addresses of random choices (containing all the random choices made during some execution) by ``t`` and partial assignments by either ``u`` or ``v``.
We denote a trace of a generative function by the tuple ``(x, t)``.
We say that two assignments ``u`` and ``t`` *agree* when they assign addresses that appear in both assignments to the same values (they can different or even disjoint sets of addresses and still agree).
A generative function is associated with a family of probability distributions ``P(t; x)`` on assignments ``t``, parameterized by arguments ``x``, and a second family of distributions ``Q(t; u, x)`` on assignments ``t`` parameterized by partial assignment ``u`` and arguments ``x``.
``Q`` is called the *internal proposal family* of the generative function, and satisfies that if ``u`` and ``t`` agree then ``P(t; x) > 0`` if and only if ``Q(t; x, u) > 0``, and that ``Q(t; x, u) > 0`` implies that ``u`` and ``t`` agree.
See the [Gen technical report](http://hdl.handle.net/1721.1/119255) for additional details.

Generative functions may also use *non-addressable random choices*, denoted ``r``.
Unlike regular (addressable) random choices, non-addressable random choices do not have addresses, and the value of non-addressable random choices is not exposed through the generative function interface.
However, the state of non-addressable random choices is maintained in the trace.
A trace that contains non-addressable random choices is denoted ``(x, t, r)``.
Non-addressable random choices manifest to the user of the interface as stochasticity in weights returned by generative function interface methods.
The behavior of non-addressable random choices is defined by an additional pair of families of distributions associated with the generative function, denoted ``Q(r; x, t)`` and ``P(r; x, t)``, which are defined for ``P(t; x) > 0``, and which satisfy ``Q(r; x, t) > 0`` if and only if ``P(r; x, t) > 0``.
For each generative function below, we describe its semantics first in the basic setting where there is no non-addressable random choices, and then in the more general setting that may include non-addressable random choices.

```@docs
initialize
project
propose
assess
force_update
fix_update
free_update
extend
backprop_params
backprop_trace
get_assmt
get_args
get_retval
get_score
```
