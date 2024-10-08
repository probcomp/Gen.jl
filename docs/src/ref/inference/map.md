# MAP Optimization

In contrast to [parameter optimization](parameter_optimization.md), which optimizes parameters of a generative function that are not associated with any prior distribution, *maximum a posteriori* (MAP)  optimization can be used to maximize the posterior probability of a selection of traced random variables:

```@docs
map_optimize
```

To use `map_optimize`, a trace of a generative function should be first created using the [`generate`](@ref) method with the appropriate observations. Users may also implement more complex optimization algorithms beyond backtracking gradient ascent by using the gradients returned by [`choice_gradients`](@ref). Note that if the selected random variables have bounded support over the real line, errors may occur during gradient-based optimization.
