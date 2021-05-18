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
The API uses tuples of the form `(gen_fn::GenerativeFunction, name::Symbol)` to identify parameters.
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

Gradient-based optimization typically involves iterating between two steps:
(i) computing gradients or estimates of gradients with respect to parameters, and
(ii) updating the value of the parameters based on the gradient estimates according to some mathematical rule.
Sometimes the optimization algorithm also has its own state that is separate from the value of the parameters and the gradient estimates.
Gradient-based optimization algorithms in Gen are implemented by **optimizers**.
Each type of parameter store provides implementations of optimizers for standard mathematical update rules.

The mathematical rules are defined in **optimizer configuration** objects.
The currently supported optimizer configurations are:
```@docs
FixedStepGradientDescent
DecayStepGradientDescent
```

The most common way to construct an optimizer is via:
```julia
optimizer = init_optimizer(conf, gen_fn)
```
which returns an optimizer that applies the mathematical rule defined by `conf` to all parameters used by `gen_fn` (even when the generative function uses parameters that are housed in multiple parameter stores).
You can also pass a parameter context keyword argument to customize the parameter store(s) that the optimizer should use.
Then, after accumulating gradients with [`accumulate_param_gradients!`](@ref), you can apply the update with:
```julia
apply_update!(optimizer)
```

The `init_optimizer` method described above constructs an optimizer that actually invokes multiple optimizers, one for each parameter store.
To add support to a parameter store type for a new optimizer configuration type, you must implement the per-parameter-store optimizer methods:

- `init_optimizer(conf, parameter_ids, store)`, which takes in an optimizer configuration object, and list of parameter IDs, and the parameter store in which to apply the updates, and returns an optimizer thata mutates the given parameter store.

- `apply_update!(optimizer)`, which takes in an a single argument (the optimizer) and applies its update rule, which mutates the value of the parameters in its parameter store (and typically also resets the values of the gradient accumulators to zero).

```@docs
init_optimizer
apply_update!
```
