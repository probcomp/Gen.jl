# Generative Function Combinators

Generative function combinators are Julia functions that take one or more generative functions as input and return a new generative function.
Generative function combinators are used to express patterns of repeated computation that appear frequently in generative models.
Some generative function combinators are similar to higher order functions from functional programming languages.
However, generative function combinators are not 'higher order generative functions', because they are not themselves generative functions (they are regular Julia functions).

## Map combinator

```@docs
Map
```

In the schematic below, the kernel is denoted ``\mathcal{G}_{\mathrm{k}}``.
```@raw html
<div style="text-align:center">
    <img src="../../images/map_combinator.png" alt="schematic of map combinator" width="50%"/>
</div>
```

For example, consider the following generative function, which makes one random choice at address `:z`:
```julia
@gen function foo(x1::Float64, x2::Float64)
    y = @trace(normal(x1 + x2, 1.0), :z)
    return y
end
```
We apply the map combinator to produce a new generative function `bar`:
```julia
bar = Map(foo)
```
We can then obtain a trace of `bar`:
```julia
(trace, _) = generate(bar, ([0.0, 0.5], [0.5, 1.0]))
```
This causes `foo` to be invoked twice, once with arguments `(0.0, 0.5)` in address namespace `1` and once with arguments `(0.5, 1.0)` in address namespace `2`.
If the resulting trace has random choices:
```
│
├── 1
│   │
│   └── :z : -0.5757913836706721
│
└── 2
    │
    └── :z : 0.7357177113395333
```
then the return value is:
```
FunctionalCollections.PersistentVector{Any}[-0.575791, 0.735718]
```


## Unfold combinator

```@docs
Unfold
```

In the schematic below, the kernel is denoted ``\mathcal{G}_{\mathrm{k}}``.
The initial state is denoted ``y_0``, the number of applications is ``n``, and the remaining arguments to the kernel not including the state, are ``z``.
```@raw html
<div style="text-align:center">
    <img src="../../images/unfold_combinator.png" alt="schematic of unfold combinator" width="70%"/>
</div>
```

For example, consider the following kernel, with state type `Bool`, which makes one random choice at address `:z`:
```julia
@gen function foo(t::Int, y_prev::Bool, z1::Float64, z2::Float64)
    y = @trace(bernoulli(y_prev ? z1 : z2), :y)
    return y
end
```
We apply the map combinator to produce a new generative function `bar`:
```julia
bar = Unfold(foo)
```
We can then obtain a trace of `bar`:
```julia
(trace, _) = generate(bar, (5, false, 0.05, 0.95))
```
This causes `foo` to be invoked five times.
The resulting trace may contain the following random choices:
```
│
├── 1
│   │
│   └── :y : true
│
├── 2
│   │
│   └── :y : false
│
├── 3
│   │
│   └── :y : true
│
├── 4
│   │
│   └── :y : false
│
└── 5
    │
    └── :y : true

```
then the return value is:
```
FunctionalCollections.PersistentVector{Any}[true, false, true, false, true]
```

## Recurse combinator

TODO: document me

```@raw html
<div style="text-align:center">
    <img src="../../images/recurse_combinator.png" alt="schematic of recurse combinatokr" width="70%"/>
</div>
```
## Switch combinator

```@docs
Switch
```

```@raw html
<div style="text-align:center">
    <img src="../../images/switch_combinator.png" alt="schematic of switch combinator" width="100%"/>
</div>
```

Consider the following constructions:

```julia
@gen function bang((grad)(x::Float64), (grad)(y::Float64))
    std::Float64 = 3.0
    z = @trace(normal(x + y, std), :z)
    return z
end

@gen function fuzz((grad)(x::Float64), (grad)(y::Float64))
    std::Float64 = 3.0
    z = @trace(normal(x + 2 * y, std), :z)
    return z
end

sc = Switch(bang, fuzz)
```

This creates a new generative function `sc`. We can then obtain the trace of `sc`:

```julia
(trace, _) = simulate(sc, (2, 5.0, 3.0))
```

The resulting trace contains the subtrace from the branch with index `2` - in this case, a call to `fuzz`:

```
│
└── :z : 13.552870875213735
```
