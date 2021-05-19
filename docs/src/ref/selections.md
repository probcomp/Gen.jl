# Selections

A *selection* represents a set of addresses of random choices.
Selections allow users to specify to which subset of the random choices in a trace a given inference operation should apply.

An address that is added to a selection indicates that either the random choice at that address should be included in the selection, or that all random choices made by a generative function traced at that address should be included.
For example, consider the following selection:
```julia
selection = select(:x, :y)
```

If we use this selection in the context of a trace of the function `baz` below, we are selecting two random choices, at addresses `:x` and `:y`:
```julia
@gen function baz()
    @trace(bernoulli(0.5), :x)
    @trace(bernoulli(0.5), :y)
end
```

If we use this selection in the context of a trace of the function `bar` below, we are actually selecting three random choices---the one random choice made by `bar` at address `:x` and the two random choices made by `foo` at addresses `:y => :z` and :y => :w`:
```julia
@gen function foo()
    @trace(normal(0, 1), :z)
    @trace(normal(0, 1), :w)
end
end

@gen function bar()
    @trace(bernoulli(0.5), :x)
    @trace(foo(), :y)
end
```

There is an abstract type for selections:
```@docs
Selection
```

There are various concrete types for selections, each of which is a subtype of [`Selection`](@ref).
Users can construct selections with the following methods:
```@docs
select
selectall
complement
```

The [`select`](@ref) method returns a selection with concrete type [`DynamicSelection`](@ref).
The [`selectall`](@ref) method returns a selection with concrete type [`AllSelection`](@ref).
The full list of concrete types of selections is shown below.
Most users need not worry about these types.
Note that only selections of type [`DynamicSelection`](@ref) are mutable (using `push!` and `set_subselection!`).
```@docs
EmptySelection
AllSelection
HierarchicalSelection
DynamicSelection
StaticSelection
```
