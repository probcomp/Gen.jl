# Generative Functions

One of the core abstractions in Gen is the **generative function**.
Generative functions are used to represent a variety of different types of probabilistic computations including generative models, inference models, custom proposal distributions, and variational approximations.

Generative functions are represented by the following abstact type:
```@docs
GenerativeFunction
```

There are various kinds of generative functions, which are represented by concrete subtypes of [`GenerativeFunction`](@ref).
For example, the [Built-in Modeling Language](@ref) allows generative functions to be constructed using Julia function definition syntax:
```julia
@gen function foo(a, b)
    if @addr(bernoulli(0.5), :z)
        return a + b + 1
    else
        return a + b
    end
end
```
Generative functions must implement the the methods of the [Generative Function Interface](@ref).
Generative functions behave like Julia functions in some respects.
For example, we can call a generative function `foo` on arguments and get an output value using regular Julia call syntax:
```julia-repl
>julia foo(2, 4)
7
```
However, generative functions are distinct from Julia functions because they support additional behaviors, described in the remainder of this section.

## Probabilistic semantics

Generative functions may use randomness, but they may not mutate externally observable state.
We represent the randomness used during an execution as a map from unique **addresses** to values, denoted ``t : A \to V`` where ``A`` is an address set and ``V`` is a set of possible values that random choices can take.
In this section, we assume that random choices are discrete to simplify notation.
We say that two random choice maps ``t`` and ``s`` **agree** if they assign the same value for all addresses that is in both of their domains.
Formally, every generative function is associated with three mathematical objects:

### Input type
The set of valid argument tuples to the function, denoted ``X``.

### Probability distribution family
A family of probability distributions ``p(t; x)`` on maps ``t`` from random choice addresses to their values, indexed by arguments ``x``, for all ``x \in X``.
Note that the distribution must be normalized:
```math
\sum_{t} p(t; x) = 1 \;\; \mbox{for all} \;\; x \in X
```
This corresponds to a requirement that the function terminate with probabability 1 for all valid arguments.

### Return value function
A (deterministic) function ``f`` that maps the tuple ``(x, t)`` of the arguments and the random choice map to the return value of the function (which we denote by ``y``).

### Internal proposal distribution family
A family of probability distributions ``q(t; x, u)`` on maps ``t`` from random choice addresses to their values, indexed by tuples ``(x, u)`` where ``u`` is a map from random choice addresses to values, and where ``x`` are the arguments to the function.
It must be normalized for all valid ``(x, u)``:
```math
\sum_{t} q(t; x, u) = 1 \;\; \mbox{for all} \;\; x \in X, u
```
Also, it must satisfy the following condition:
```math
p(t; x) > 0 \mbox{ if and only if } q(t; x, u) > 0 \mbox{ for all } u \mbox{ where } u \mbox{ and } t \mbox{ agree }
```
Finally, we require that:
```math
q(t; x, u) > 0 \mbox{ implies that } u \mbox{ and } t \mbox{ agree }.
```


## Execution traces

An **execution trace** (or just *trace*) is a record of an execution of a generative function.
There is no abstract type representing all traces.
Different concrete types of generative functions use different data structures and different Jula types for their traces.
The trace type that a generative function uses is the second type parameter of the [`GenerativeFunction`](@ref) abstract type.

A trace of a generative function can be produced using [`initialize`](@ref):
```julia
(trace, weight) = initialize(foo, (2, 4))
```

The trace contains various information about the execution, including:

The arguments to the function, which is retrieved with [`get_args`](@ref):
```julia-repl
>julia get_args(trace)
(2, 4)
```

The return value of the function, which is retrieved with [`get_retval`](@ref):
```julia-repl
>julia get_retval(trace)
7
```

The map from addresses of random choices to their values, which is retrieved with [`get_assmt`](@ref), and has abstract type [`Assignment`](@ref).
```julia-repl
>julia println(get_assmt(trace))
│
└── :z : false
```

The log probability that the random choices took the values they did for the arguments, available with [`get_score`](@ref):
```julia-repl
>julia get_score(trace)
-0.6931471805599453
```

A reference to the generative function that was executed, which is retrieved with [`get_gen_fn`](@ref):
```julia-repl
>julia foo === get_gen_fn(trace)
true
```

## Trace update methods

There are several methods that take a trace of a generative function as input and return a new trace of the generative function based on adjustments to the execution history of the function:

- [`force_update`](@ref)
- [`fix_update`](@ref)
- [`free_update`](@ref)
- [`extend`](@ref)

In addition to the input trace, and other arguments that indicate how to adjust the trace, each of these methods also accepts an **args** argument and an **argdiff** argument.
The args argument contains the new arguments to the generative function, which may differ from the previous arguments to the generative function (which can be retrieved by applying [`get_args`](@ref) to the previous trace).
In many cases, the adjustment to the execution specified by the other arguments to these methods is 'small' and only effects certain parts of the computation.
Therefore, it is often possible to generate the new trace and the appropriate log probability ratios required for these methods without revisiting every state of the computation of the generative function.
To enable this, the argdiff argument provides additional information about the *difference* between the previous arguments to the generative function, and the new arguments.
This argdiff information permits the implementation of the update method to avoid inspecting the entire argument data structure to identify which parts were updated.

## Differentiable programming

Generative functions may support computation of gradients with respect to (i) all or a subset of its arguments, (ii) its **trainable parameters**, and (iii) the value of certain random choices.
The set of elements (either arguments, trainable parameters, or random choices) for which gradients are available is called the *gradient source set*.
A generative function statically reports whether or not it is able to compute gradients with respect to each of its arguments, through the function `has_argument_grads`.
Let ``x_G`` denote the set of arguments for which the generative function does support gradient computation.
Similarly, a generative function supports gradients with respect the value of random choices made at all or a subset of addresses.
If the return value of the function is conditionally independent of each element in the gradient source set given the other elements in the gradient source set and values of all other random choices, for all possible traces of the function, then the generative function requires a *return value gradient* to compute gradients with respect to elements of the gradient source set.
This static property of the generative function is reported by `accepts_output_grad`.

## Trainable parameters

## Non-addressed randomness

Generative functions may also use **non-addressable randomness**, which is not returned by [`get_assmt`](@ref).
However, the state of non-addressable random choices *is* maintained by the trace internally.
We denote non-addressable randomness by ``r``.
The probabilistic semantics are extended for a generative function with non-addressable randomness as follows:

### Input type
No extension necessary

### Probability distribution family
A family of probability distributions ``p(t, r; x)`` that is normalized for all ``x \in X``, that factors according to:
```math
p(t, r; x) = p(t; x) p(r; t, x)
```

### Return value function
The return value *cannot* be a function of the non-addressable randomness.
It remains a function ``f`` on tuples ``(x, t)``.

### Internal proposal distribution family
A family of distributions ``q(t, r; x, u)`` that factors according to:
```math
q(t, r; x, u) = q(t; x, u) q(r; x, t)
```
where ``q(t; x, u)`` satisfies the conditions stated above.
Note that the distribution on internal randomness does not depend on ``u`` in this factorization.


### Generative Function Interface

```@docs
has_argument_grads
accepts_output_grad
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
get_assmt
get_args
get_retval
get_score
get_gen_fn
get_params
```
