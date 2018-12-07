# Gen Documentation

## Assignment

### Static Assignment

A `StaticAssignment` contains only symbols as its keys for leaf nodes and for internal nodes.
A `StaticAssignment` has type parameters`R` and `T` that are tuples of `Symbol`s that are the keys of the leaf nodes and internal nodes respectively, so that code can be generated that is specialized to the particular set of keys in the trie:

```julia
struct StaticAssignment{R,S,T,U} <: Assignment
    leaf_nodes::NamedTuple{R,S}
    internal_nodes::NamedTuple{T,U}
end 
```

A `StaticAssignment` with leaf symbols `:a` and `:b` and internal key `:c` can be constructed using syntax like:
```julia
trie = StaticAssignment((a=1, b=2), (c=inner_trie,))
```


## Generative Function Interface

Generative functions implement the *generative function interface*, which is a set of methods that involve the execution traces and probabilistic behavior of generative functions.
Mathematically, a generative function is associated with a family of probability distributions P(u; x) on assignments (u), parameterized by arguments (x) to the function.
A generative function is also associated with a second family of probability distributions Q(u; v, x) on assignments (u), parametrized by an assignment (v) and arguments (x) to the function.
Q is called the *internal proposal distribution* of the generative function.
Q satisfies the requirement that an assignment u sampled from Q(.; v, x) with nonzero probability must agree with the assignment v on all addresses that appear in both.
Below, we use 't' to denote to a *complete assignment* (an assignment containing all random choices in some execution trace) for some arguments, which we denote 'x'.
See the [Gen technical report](http://hdl.handle.net/1721.1/119255) for additional details.

```@docs
initialize
project
propose
assess
force_update
fix_update
free_update
extend
backprop_params
backprop_trace
```

## Distributions

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

Distribution values are also callable, which is a syntactic sugar with the same behavior of calling `random`:

```julia
bernoulli(0.5) # identical to random(bernoulli, 0.5) and random(Bernoulli(), 0.5)
```

### Gradients of Distributions

Distributions may also implement `logpdf_grad`, which returns the gradient of the log probability (density) with respect to the random sample and the parameters, as a tuple:

```julia
(grad_sample, grad_mu, grad_std) = logpdf_grad(normal, 1.324, 0.0, 1.0)
```

The partial derivative of the log probability (density) with respect to the random sample, or one of the parameters, might not always exist.
Distributions indicate which partial derivatives exist using the methods `has_output_grad` and `has_argument_grads`:

```julia
has_output_grad(::Normal) = true
has_argument_grads(::Normal) = (true, true)
```

If a particular partial derivative does not exist, that field of the tuple returned by `logpdf_grad` should be `nothing`.


### Built-In Distributions

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
```
