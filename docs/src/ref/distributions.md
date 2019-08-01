## Probability Distributions

## Built-In Distributions

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
exponential
```

## Defining New Distributions

Probability distributions are singleton types whose supertype is `Distribution{T}`, where `T` indicates the data type of the random sample.

```julia
abstract type Distribution{T} end
```

By convention, distributions have a global constant lower-case name for the singleton value.
For example:

```julia
struct Bernoulli <: Distribution{Bool} end
const bernoulli = Bernoulli()
```

Distributions must implement two methods, `random` and `logpdf`.

`random` returns a random sample from the distribution:

```julia
x::Bool = random(bernoulli, 0.5)
x::Bool = random(Bernoulli(), 0.5)
```

`logpdf` returns the log probability (density) of the distribution at a given value:

```julia
logpdf(bernoulli, false, 0.5)
logpdf(Bernoulli(), false, 0.5)
```

Distribution values should also be callable, which is a syntactic sugar with the same behavior of calling `random`:

```julia
bernoulli(0.5) # identical to random(bernoulli, 0.5) and random(Bernoulli(), 0.5)
```

A new Distribution type must implement the following methods:
```@docs
random
logpdf
has_output_grad
logpdf_grad
```
A new Distribution type must also implement [`has_argument_grads`](@ref).
