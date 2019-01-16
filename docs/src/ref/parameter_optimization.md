# Optimizing Static Parameters

Gradient-based optimization of the static parameters of generative functions is based on interleaving (1) accumulation of gradients with respect to static parameters using `backprop_params` with (2) updates to the static parameters that read from the gradients, mutate the value of the static parameters, and reset the gradients to zero.

Gen has built-in support for the following configuration types of updates:
```@docs
GradientDescent
ADAM
```
Parameter updates are constructed by combining a configuration value with the set of static parameters that it should be applied to:
```@docs
ParamUpdate
```
Finally, an update is applied with:
```@docs
apply!
```

