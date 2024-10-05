# [Adding New Distributions](@id custom_distributions_howto)

In addition to the built-in distributions, mixture distributions, and product distributions,
Gen provides two primary ways of adding new distributions:

## [Defining New Distributions Inline with the `@dist` DSL](@id dist_dsl_howto)

The `@dist` DSL allows users to concisely define a distribution, as long as
that distribution can be expressed as a certain type of deterministic
transformation of an existing distribution:

```julia
@dist name(arg1, arg2, ..., argN) = body
```
or
```julia
@dist function name(arg1, arg2, ..., argN)
    body
end
```

Here `body` is ordinary Julia code, with the constraint that `body` must
contain exactly one random choice.  The resulting distribution is called `name`,
parameterized by `arg1, ..., argN`, and represents a distribution over
_return values_ of `body`. 

### Common Use Cases

The `@dist` DSL makes it easy to implement labeled uniform or categorical
distributions, instead of having to use integers to refer to categories:

```julia
"""Labeled uniform distribution"""
@dist labeled_uniform(labels) = labels[uniform_discrete(1, length(labels))]

"""Labeled categorical distribution"""
@dist labeled_categorical(labels, probs) = labels[categorical(probs)]
```

Note however that there is a slight overhead incurred by having to detect the 
possibility of duplicate labels.

It is also possible to implement a distribution that takes a Boolean input 
and flips it with some probability:

```julia
"Bit-flip distribution."
@dist bit_flip(x::Bool, p::Float64) = bernoulli((1-x) * p + x * (1-p))
```

Finally, it is possible to distributions that are defined in terms of shifting, 
scaling, exponentiation, or taking the logarithm of another distribution:

```julia
"Symmetric Binomial distribution."
@dist sym_binom(mean::Int, scale::Int) = binom(2*scale, 0.5) - scale + mean

"Shifted geometric distribution."
@dist shifted_geometric(p::Real, shift::Int) = geometric(p) + shift

"Log-normal distribution."
@dist log_normal(mu::Real, sigma::Real) = exp(normal(mu, sigma))

"Gumbel distribution."
@dist gumbel(mu::Real, beta::Real) = mu - beta * log(0.0 - log(uniform(0, 1)))
```

### Restrictions

There are a number of restrictions imposed by the `@dist` DSL, which are
explained further in the [reference docmumentation](@ref dist_dsl).
Most importantly, only a limited set of deterministic transformations are currently
supported (`+`, `-`, `*`, `/`, `exp`, `log`, `getindex`), and only *one* random
choice can be used in the body of the definition.

## Defining New Distributions From Scratch

For distributions that cannot be expressed in the `@dist` DSL, users can define
a custom distribution by defining an (ordinary Julia) subtype of
`Gen.Distribution` and implementing the methods of the [Distribution API](@ref
distributions).  This method requires more custom code than using the
`@dist` DSL, but also affords more flexibility: arbitrary user-defined logic
for sampling, PDF evaluation, etc.

Probability distributions are singleton types whose supertype is `Distribution{T}`, where `T` indicates the data type of the random sample.

```julia
abstract type Distribution{T} end
```

A new `Distribution` type must implement the following methods:

- [`random`](@ref)

- [`logpdf`](@ref)

- [`logpdf_grad`](@ref)

- [`has_argument_grads`](@ref)

- [`has_output_grad`](@ref)

- [`is_discrete`](@ref)

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

For example, this can be done by adding a method definition for `Bernoulli`:

```julia
(::Bernoulli)(prob) = random(Bernoulli(), prob)
```
