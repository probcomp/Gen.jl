# [Custom Generative Functions](@id custom_gen_fns)

Gen provides scaffolding for custom (deterministic) generative functions that make use of either incremental computation or custom gradient updates.

```@docs
CustomDetermGF
CustomDetermGFTrace
```

## Custom Incremental Computation

A [`CustomUpdateGF`](@ref) is a generative function that allows for easier implementation of custom incremental computation.

```@docs
CustomUpdateGF
apply_with_state
update_with_state
num_args
```

## Custom Gradient Computations

A [`CustomGradientGF`](@ref) is a generative function that allows for easier implementation of custom gradients computations and updates.

```@docs
CustomGradientGF
apply
gradient
```
