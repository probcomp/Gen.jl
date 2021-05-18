# Optimizing Trainable Parameters

## Parameter stores

Multiple traces of a generative function typically reference the same trainable parameters of the generative function, which are stored outside of the trace in a **parameter store**.
Different types of generative functions may use different types of parameter stores.
For example, the [`JuliaParameterStore`](@ref) (discussed below) stores parameters as Julia values in the memory of the Julia runtime process.
Other types of parameter stores may store parameters in GPU memory, in a filesystem, or even remotely.

When generating a trace of a generative function with [`simulate`](@ref) or [`generate`](@ref), we may pass in an optional **parameter context**, which is a `Dict` that provides information about which parameter store(s) in which to look up the value of parameters.
A generative function obtains a reference to a specific type of parameter store by looking up its key in the parameter context.

If you are just learning Gen, and are only using the built-in modeling language to write generative functions, you can ignore this complexity, because there is a [`default_julia_parameter_store`](@ref) and a default parameter context [`default_parameter_context`](@ref) that points to this default Julia parameter store that will be used if a parameter context is not provided in the call to `simulate` and `generate`.
```@docs
default_parameter_context
default_julia_parameter_store
```

## Julia parameter store

Parameters declared using the `@param` keyword in the built-in modeling language are stored in a type of parameter store called a [`JuliaParameterStore`](@ref).
A generative function can obtain a reference to a `JuliaParameterStore` by looking up the key [`JULIA_PARAMETER_STORE_KEY`](@ref) in a parameter context.
This is how the built-in modeling language implementation finds the parameter stores to use for `@param`-declared parameters.
Note that if you are defining your own [custom generative functions](@ref #Custom-generative-functions), you can also use a [`JuliaParameterStore`](@ref) (including the same parameter store used to store parameters of built-in modeling language generative functions) to store and optimize your trainable parameters.

Different types of parameter stores provide different APIs for reading, writing, and updating the values of parameters and gradient accumulators for parameters.
The `JuliaParameterStore` API is given below.
(Note that most user learning code only needs to use [`init_parameter!`](@ref), as the other API functions are called by [Optimizers](@ref) which are discussed below.)

```@docs
JuliaParameterStore
init_parameter!
increment_gradient!
reset_gradient!
get_parameter_value
get_gradient
JULIA_PARAMETER_STORE_KEY
```

### Multi-threaded gradient accumulation

Note that the [`increment_gradient!`](@ref) call is thread-safe, so that multiple threads can concurrently increment the gradient for the same parameters. This is helpful for parallelizing gradient computation for a batch of traces within stochastic gradient descent learning algorithms.

## Optimizers

TODO

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
The set of possible update configurations is described in [Update configurations](@ref).
An update is applied with:
```@docs
apply!
```

## Update configurations

Gen has built-in support for the following types of update configurations.
```@docs
FixedStepGradientDescent
GradientDescent
ADAM
```
For adding new types of update configurations, see [Optimizing Trainable Parameters (Internal)](@ref optimizing-internal).
