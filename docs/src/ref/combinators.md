# Generative Function Combinators

Generative function combinators are Julia functions that take one or more generative functions as input and return a new generative function.
Generative function combinators are used to express patterns of repeated computation that appear frequently in generative models.
Some generative function combinators are similar to higher order functions from functional programming languages.
However, generative function combinators are not 'higher order generative functions', because they are not themselves generative functions (they are regular Julia functions).

## Map combinator

The *map combinator* takes a generative function as input, and returns a generative function that applies the given generative function independently to a vector of arguments.
The returned generative function has one argument with type `Vector{T}` for each argument of type `T` of the input generative function.
The length of each argument, which must be the same for each argument, determines the number of times the input generative function is called (N).
Each call to the input function is made under address namespace i for i=1..N.
The return value of the returned function has type `Vector{T}` where `T` is the type of the return value of the input function.
The map combinator is similar to the 'map' higher order function in functional programming, except that the map combinator returns a new generative function that must then be separately applied.

For example, consider the following generative function, which makes one random choice at address `:z`:
```julia
@gen function foo(x::Float64, y::Float64)
    @addr(normal(x + y, 1.0), :z)
end
```

We apply the map combinator to produce a new generative function `bar`:
```julia
bar = Map(foo)
```

We can then obtain a trace of `bar`:
```julia
xs = [0.0, 0.5]
ys = [0.5, 1.0]
(trace, _) = initialize(bar, (xs, ys))
```

This causes `foo` to be invoked twice, once with arguments `(xs[1], ys[1])` in address namespace `1` and once with arguments `(xs[2], ys[2])` in address namespace `2`.
The resulting trace has random choices at addresses `1 => :z` and `2 => :z`.


## Unfold combinator

## Recurse combinator


