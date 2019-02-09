# Simple Generative Functions

Standard probability distributions, for which it is possible to sample random variates, as well as evaluate the log probability (density) pointwise, are represented as *Simple Generative Functions*.
Unlike generative functions, simple generative functions do not have an address space for random choices, and they do not have a trace that is separate from their return value.
Note that the return value of a simple generative function can have arbitrary type including a scalar, or vector, etc.

Note that every simple generative function can in principle also be 'boxed' as a regular generative function that makes one random choice at one address (e.g. `:output`).
However, this would be more verbose and would introduce additional overhead.

## Built-In Simple Generative Functions

```@docs
bernoulli
normal
mvnormal
gamma
inv_gamma
beta
categorical
uniform
uniform_discrete
poisson
piecewise_uniform
beta_uniform
```

## Defining Simple Generative Functions

Simple generative functions are values with supertype:
```@docs
SimpleGenerativeFunction
```

By convention, the built-in simple generative functions have a global constant lower-case name for the singleton value.
For example:
```julia
struct Bernoulli <: SimpleGenerativeFunction{Bool} end
const bernoulli = Bernoulli()
```

Simple generative functions must implement the following methods:
```@docs
random
logpdf
```

Simple generative functions must also implement the method [`has_argument_grads`](@ref), as well as:
```@docs
has_output_grad
logpdf_grad
```

Simple generative functions values should also be callable, which is a syntactic sugar with the same behavior of calling `random`:
```julia
bernoulli(0.5) # identical to random(bernoulli, 0.5) and random(Bernoulli(), 0.5)
```
