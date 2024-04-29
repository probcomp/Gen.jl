# Generative Combinators

Generative function combinators are Julia functions that take one or more generative functions as input and return a new generative function. Generative function combinators are used to express patterns of repeated computation that appear frequently in generative models. Some generative function combinators are similar to higher order functions from functional programming languages.

## Map combinator

In the schematic below, the kernel is denoted ``\mathcal{G}_{\mathrm{k}}``.
```@raw html
<div style="text-align:center">
    <img src="../../images/map_combinator.png" alt="schematic of map combinator" width="50%"/>
</div>
```

For example, consider the following generative function, which makes one random choice at address `r^2`:
```@example map_combinator
using Gen
@gen function foo(x, y, z)
    r ~ normal(x^2 + y^2 + z^2, 1.0)
    return r
end
```
We apply the map combinator to produce a new generative function `bar`:
```@example map_combinator
bar = Map(foo)
```
We can then obtain a trace of `bar`:
```@example map_combinator
trace, _ = generate(bar, ([0.0, 0.5], [0.5, 1.0], [1.0, -1.0]))
trace
```
This causes `foo` to be invoked twice, once with arguments `(0.0, 0.5, 1.0)` in address namespace `1` and once with arguments `(0.5, 1.0, -1.0)` in address namespace `2`.

```@example map_combinator
get_choices(trace)
```
If the resulting trace has random choices:
then the return value is:

```@example map_combinator
get_retval(trace)
```

## Unfold combinator

In the schematic below, the kernel is denoted ``\mathcal{G}_{\mathrm{k}}``.
The initial state is denoted ``y_0``, the number of applications is ``n``, and the remaining arguments to the kernel not including the state, are ``z``.
```@raw html
<div style="text-align:center">
    <img src="../../images/unfold_combinator.png" alt="schematic of unfold combinator" width="70%"/>
</div>
```

For example, consider the following kernel, with state type `Bool`, which makes one random choice at address `:z`:
```@example unfold_combinator
using Gen
@gen function foo(t::Int, y_prev::Bool, z1::Float64, z2::Float64)
    y = @trace(bernoulli(y_prev ? z1 : z2), :y)
    return y
end
```
We apply the map combinator to produce a new generative function `bar`:
```@example unfold_combinator
bar = Unfold(foo)
```
We can then obtain a trace of `bar`:
```@example unfold_combinator
trace, _ = generate(bar, (5, false, 0.05, 0.95))
trace
```
This causes `foo` to be invoked five times.
The resulting trace may contain the following random choices:
```@example unfold_combinator
get_choices(trace)
```
then the return value is:
```@example unfold_combinator
get_retval(trace)
```

## Switch combinator

```@raw html
<div style="text-align:center">
    <img src="../../images/switch_combinator.png" alt="schematic of switch combinator" width="100%"/>
</div>
```

Consider the following constructions:

```@setup switch_combinator
using Gen
```

```@example switch_combinator
@gen function line(x)
    z ~ normal(3*x+1,1.0)
    return z
end

@gen function outlier(x)
    z ~ normal(3*x+1, 10.0)
    return z
end

switch_model = Switch(line, outlier)
```

This creates a new generative function `switch_model` whose arguments take the form `(branch, args...)`. By default,
branch is an integer indicating which generative function to execute. For example, branch `2` corresponds to `outlier`:

```@example switch_combinator
trace = simulate(switch_model, (2, 5.0))
get_choices(trace)
```
