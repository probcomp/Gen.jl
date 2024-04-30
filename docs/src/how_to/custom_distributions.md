# How to write custom Gen distribution

## [Custom distributions](@id custom_distributions)

Users can extend Gen with new probability distributions, which can then be used
to make random choices within generative functions. Simple transformations of
existing distributions can be created using the [`@dist` DSL](@ref dist_dsl).
For arbitrary distributions, including distributions that cannot be expressed
in the `@dist` DSL, users can define a custom distribution by implementing
Gen's Distribution interface directly, as defined below.

Probability distributions are singleton types whose supertype is `Distribution{T}`, where `T` indicates the data type of the random sample.

```julia
abstract type Distribution{T} end
```

A new `Distribution` type must implement the following methods:

- [`random`](@ref)

- [`logpdf`](@ref)

- [`has_output_grad`](@ref)

- [`logpdf_grad`](@ref)

- [`has_argument_grads`](@ref)


By convention, distributions have a global constant lower-case name for the singleton value.
For example:

```julia
struct Bernoulli <: Distribution{Bool} end
const bernoulli = Bernoulli()
```
Distribution values should also be callable, which is a syntactic sugar with the same behavior of calling `random`:

```julia
bernoulli(0.5) # identical to random(bernoulli, 0.5) and random(Bernoulli(), 0.5)
```

## [Custom Generative Functions](@id custom_generative_functions)

We recommend the following steps for implementing a new type of generative function, and also looking at the implementation for the [`DynamicDSLFunction`](@ref) type as an example.

##### Define a trace data type
```julia
struct MyTraceType <: Trace
    ..
end
```

##### Decide the return type for the generative function
Suppose our return type is `Vector{Float64}`.

##### Define a data type for your generative function
This should be a subtype of [`GenerativeFunction`](@ref), with the appropriate type parameters.
```julia
struct MyGenerativeFunction <: GenerativeFunction{Vector{Float64},MyTraceType}
..
end
```
Note that your generative function may not need to have any fields.
You can create a constructor for it, e.g.:
```
function MyGenerativeFunction(...)
..
end
```

##### Decide what the arguments to a generative function should be
For example, our generative functions might take two arguments, `a` (of type `Int`) and `b` (of type `Float64`).
Then, the argument tuple passed to e.g. [`generate`](@ref) will have two elements.

NOTE: Be careful to distinguish between arguments to the generative function itself, and arguments to the constructor of the generative function.
For example, if you have a generative function type that is parametrized by, for example, modeling DSL code, this DSL code would be a parameter of the generative function constructor.

##### Decide what the traced random choices (if any) will be
Remember that each random choice is assigned a unique address in (possibly) hierarchical address space.
You are free to design this address space as you wish, although you should document it for users of your generative function type.

##### Implement methods of the Generative Function Interface

At minimum, you need to implement the following methods:

- [`simulate`](@ref)

- [`has_argument_grads`](@ref)

- [`accepts_output_grad`](@ref)

- [`get_args`](@ref)

- [`get_retval`](@ref)

- [`get_choices`](@ref)

- [`get_score`](@ref)

- [`get_gen_fn`](@ref)

- [`project`](@ref)

If you want to use the generative function within models, you should implement:

- [`generate`](@ref)

If you want to use MCMC on models that call your generative function, then implement:

- [`update`](@ref)

- [`regenerate`](@ref)

If you want to use gradient-based inference techniques on models that call your generative function, then implement:

- [`choice_gradients`](@ref)

- [`update`](@ref)

If your generative function has trainable parameters, then implement:

- [`accumulate_param_gradients!`](@ref)


