# Parameter Optimization

Gen provides support for gradient-based optimization of the trainable parameters of generative functions, e.g. for maximixum likelihood learning, expectation-maximization, or empirical Bayes.

Trainable parameters of generative functions are initialized differently depending on the type of generative function.
Trainable parameters of the built-in modeling language are initialized with [`init_param!`](@ref).

Gradient-based optimization of the trainable parameters of generative functions is based on interleaving two steps:

- Incrementing gradient accumulators for trainable parameters by calling [`accumulate_param_gradients!`](@ref) on one or more traces.

- Updating the value of trainable parameters and resetting the gradient accumulators to zero, by calling [`apply!`](@ref) on a *parameter update*, as described below.

## Parameter update

A *parameter update* reads from the gradient accumulators for certain trainable parameters, updates the values of those parameters, and resets the gradient accumulators to zero.

A paramter update is constructed by combining an *update configuration* with the set of trainable parameters to which the update should be applied:

```@docs
ParamUpdate
```
The set of possible update configurations is described in [Update configurations](@ref update_configurations).An update is applied with:

```@docs
apply!
```

## [Update configurations](@id update_configurations)

Gen has built-in support for the following types of update configurations.
```@docs
FixedStepGradientDescent
GradientDescent
ADAM
```

Custom update configurations should implement the following methods:

```@docs
Gen.init_update_state
Gen.apply_update!
```

## Training generative functions

The [`train!`](@ref) method can be used to train the parameters of a generative function to maximize the likelihood of a dataset defined by a `data_generator`:

```@docs
train!
```
