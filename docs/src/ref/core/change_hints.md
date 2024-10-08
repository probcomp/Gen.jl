# Change Hints

Gen defines a system of *change hints*, or [`Diff`](@ref) types, in order to
pass information to and from generative functions about whether their arguments
and return values have changed (and how). 

```@docs
Diff
```

Change hints are used to support incremental computation in
[generative function combinators](@ref combinators) and the
[static modeling language](@ref sml).  The most important change hints are
[`NoChange`](@ref) and [`UnknownChange`](@ref), which respectively indicate that
a value has not changed and that a value may have changed in an unknown way.

```@docs
NoChange
UnknownChange
```

A number of other change hints are also implemented in Gen, such as `IntDiff`, 
`VectorDiff`, and `SetDiff` and `DictDiff`, which support incremental
computation for certain operations when applied to [`Diffed`](@ref) values. 
These change hints are not documented, and are currently considered an 
implementation detail.

```@docs
Diffed
```
