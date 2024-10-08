# [Generative Function Interface](@id gfi)

One of the core abstractions in Gen is the **generative function**. The interface for interacting with generative functions is called the **generative function interface (GFI)** .
Generative functions are used to represent a variety of different types of probabilistic computations including generative models, inference models, custom proposal distributions, and variational approximations.

## Introduction

There are various kinds of generative functions, which are represented by concrete subtypes of [`GenerativeFunction`](@ref).

For example, the [Built-in Modeling Language](@ref dml) allows generative functions to be constructed using Julia function definition syntax:
```@example gfi_tutorial
using Gen # hide
@gen function foo(a, b=0)
    if @trace(bernoulli(0.5), :z)
        return a + b + 1
    else
        return a + b
    end
end;
```
Users can also extend Gen by implementing their own [custom generative functions](@ref custom_gen_fns), which can be new modeling languages, or just specialized optimized implementations of a fragment of a specific model.

Generative functions behave like Julia functions in some respects.
For example, we can call a generative function `foo` on arguments and get an output value using regular Julia call syntax:
```@example gfi_tutorial
foo(2, 4)
```

However, generative functions are distinct from Julia functions because they support additional behaviors, described in the remainder of this section.


## Mathematical concepts

Generative functions represent computations that accept some arguments, may use randomness internally, return an output, and cannot mutate externally observable state.
We represent the randomness used during an execution of a generative function as a **choice map** from unique **addresses** to values of random choices, denoted ``t : A \to V`` where ``A`` is a finite (but not a priori bounded) address set and ``V`` is a set of possible values that random choices can take.
In this section, we assume that random choices are discrete to simplify notation.
We say that two choice maps ``t`` and ``s`` **agree** if they assign the same value for any address that is in both of their domains.

Generative functions may also use **non-addressable randomness**, which is not included in the map ``t``.
We denote non-addressable randomness by ``r``.
Untraced randomness is useful for example, when calling black box Julia code that implements a randomized algorithm.

The observable behavior of every generative function is defined by the following mathematical objects:

### Input type
The set of valid argument tuples to the function, denoted ``X``.

### Probability distribution family
A family of probability distributions ``p(t, r; x)`` on maps ``t`` from random choice addresses to their values, and non-addressable randomness ``r``, indexed by arguments ``x``, for all ``x \in X``.
Note that the distribution must be normalized:
```math
\sum_{t, r} p(t, r; x) = 1 \;\; \text{for all} \;\; x \in X
```
This corresponds to a requirement that the function terminate with probabability 1 for all valid arguments.
We use ``p(t; x)`` to denote the marginal distribution on the map ``t``:
```math
p(t; x) := \sum_{r} p(t, r; x)
```
And we denote the conditional distribution on non-addressable randomness ``r``, given the map ``t``, as:
```math
p(r | t; x) := p(t, r; x) / p(t; x)
```

### Return value function
A (deterministic) function ``f`` that maps the tuple ``(x, t)`` of the arguments and the choice map to the return value of the function (which we denote by ``y``).
Note that the return value cannot depend on the non-addressable randomness.

### Auxiliary state
Generative functions may expose additional **auxiliary state** associated with an execution, besides the choice map and the return value.
This auxiliary state is a function ``z = h(x, t, r)`` of the arguments, choice map, and non-addressable randomness.
Like the choice map, the auxiliary state is indexed by addresses.
We require that the addresses of auxiliary state are disjoint from the addresses in the choice map.
Note that when a generative function is called within a model, the auxiliary state is not available to the caller.
It is typically used by inference programs, for logging and for caching the results of deterministic computations that would otherwise need to be reconstructed.

### Internal proposal distribution family
A family of probability distributions ``q(t; x, u)`` on maps ``t`` from random choice addresses to their values, indexed by tuples ``(x, u)`` where ``u`` is a map from random choice addresses to values, and where ``x`` are the arguments to the function.
It must satisfy the following conditions:
```math
\sum_{t} q(t; x, u) = 1 \;\; \text{for all} \;\; x \in X, u
```
```math
p(t; x) > 0 \text{ if and only if } q(t; x, u) > 0 \text{ for all } u \text{ where } u \text{ and } t \text{ agree }
```
```math
q(t; x, u) > 0 \text{ implies that } u \text{ and } t \text{ agree }.
```
There is also a family of probability distributions ``q(r; x, t)`` on non-addressable randomness, that satisfies:
```math
q(r; x, t) > 0 \text{ if and only if } p(r | t, x) > 0
```


## Traces

An **execution trace** (or just *trace*) is a record of an execution of a generative function.
Traces are the primary data structures manipulated by Gen inference programs.
There are various methods for producing, updating, and inspecting traces.
Traces contain:

- the arguments to the generative function

- the choice map

- the return value

- auxiliary state

- other implementation-specific state that is not exposed to the caller or user of the generative function, but is used internally to facilitate e.g. incremental updates to executions and automatic differentiation

- any necessary record of the non-addressable randomness


Different concrete types of generative functions use different data structures and different Julia types for their traces, but traces are subtypes of [`Trace`](@ref).

The concrete trace type that a generative function uses is the second type parameter of the [`GenerativeFunction`](@ref) abstract type.
For example, the trace type of [`DynamicDSLFunction`](@ref) is `DynamicDSLTrace`.

A generative function can be executed to produce a trace of the execution using [`simulate`](@ref):
```julia
trace = simulate(foo, (a, b))
```
A traced execution that satisfies constraints on the choice map can be generated using [`generate`](@ref):
```julia
trace, weight = generate(foo, (a, b), choicemap((:z, false)))
```

There are various methods for inspecting traces, including:

- [`get_args`](@ref) (returns the arguments to the function)

- [`get_retval`](@ref) (returns the return value of the function)

- [`get_choices`](@ref) (returns the choice map)

- [`get_score`](@ref) (returns the log probability that the random choices took the values they did)

- [`get_gen_fn`](@ref) (returns a reference to the generative function)

You can also access the values in the choice map and the auxiliary state of the trace by passing the address to [`Base.getindex`](@ref).
For example, to retrieve the value of random choice at address `:z`:
```julia
z = trace[:z]
```

When a generative function has default values specified for trailing arguments, those arguments can be left out when calling [`simulate`](@ref), [`generate`](@ref), and other functions provided by the generative function interface. The default values will automatically be filled in:
```@example gfi_tutorial
trace = simulate(foo, (2,));
get_args(trace)
```

## Updating traces

It is often important to incrementally modify the trace of a generative function (e.g. within MCMC, numerical optimization, sequential Monte Carlo, etc.).
In Gen, traces are **functional data structures**, meaning they can be treated as immutable values.
There are several methods that take a trace of a generative function as input and return a new trace of the generative function based on adjustments to the execution history of the function.
We will illustrate these methods using the following generative function:
```@example gfi_tutorial
@gen function bar()
    val = @trace(bernoulli(0.3), :a)
    if @trace(bernoulli(0.4), :b)
        val = @trace(bernoulli(0.6), :c) && val
    else
        val = @trace(bernoulli(0.1), :d) && val
    end
    val = @trace(bernoulli(0.7), :e) && val
    return val
end
```

Suppose we have a trace (`trace`) of `bar` with initial choices:
```@example gfi_tutorial
trace, _ = generate(bar, (), choicemap(:a=>false, :b=>true, :c=>false, :e=>true)) # hide
nothing # hide
```

```@example gfi_tutorial
get_choices(trace) # hide
```

Note that address `:d` is not present because the branch in which `:d` is sampled was not taken because random choice `:b` had value `true`.

### Update
The [`update`](@ref) method takes a trace and generates an adjusted trace that is consistent with given changes to the arguments to the function, and changes to the values of random choices made.

**Example.**
Suppose we run [`update`](@ref) on the example `trace`, with the following constraints:

```@example gfi_tutorial
choicemap(:b=>false, :d=>true) #hide
```

```@example gfi_tutorial
constraints = choicemap((:b, false), (:d, true))
(new_trace, w, _, discard) = update(trace, (), (), constraints)
```
Then `get_choices(new_trace)` will be:

```@example gfi_tutorial
get_choices(new_trace)
```

and `discard` will be:

```@example gfi_tutorial
discard
```

Note that the discard contains both the previous values of addresses that were overwritten, and the values for addresses that were in the previous trace but are no longer in the new trace.
The weight (`w`) is computed as:
```math
p(t; x) = 0.7 × 0.4 × 0.4 × 0.7 = 0.0784\\
p(t'; x') = 0.7 × 0.6 × 0.1 × 0.7 = 0.0294\\
w = \log p(t'; x')/p(t; x) = \log 0.0294/0.0784 = \log 0.375
```

**Example.**
Suppose we run [`update`](@ref) on the example `trace`, with the following constraints, which *do not* contain a value for `:d`:

```@example gfi_tutorial
choicemap(:b=>false) # hide
```

```@example gfi_tutorial
constraints = choicemap((:b, false))
(new_trace, w, _, discard) = update(trace, (), (), constraints)
```

Since `b` is constrained to `false`, the updated trace must now sample at address `d` (note address `e` remains fixed). There two possibilities for `get_choices(new_trace)`:

The first choicemap:
```@example gfi_tutorial
choicemap(:a=>false, :b=>false, :d=>true, :e=>true) # hide
```
occurs with probability 0.1. The second:

```@example gfi_tutorial
choicemap(:a=>false, :b=>false, :d=>false, :e=>true) # hide
```

occurs with probability 0.9.
Also, `discard` will be:
```@example gfi_tutorial
discard # hide
```

If the former case occurs and `:d` is assigned to `true`, then the weight (`w`) is computed as:
```math
p(t; x) = 0.7 × 0.4 × 0.4 × 0.7 = 0.0784\\
p(t'; x') = 0.7 × 0.6 × 0.1 × 0.7 = 0.0294\\
q(t'; x', t + u) = 0.1\\
w = \log p(t'; x')/(p(t; x) q(t'; x', t + u)) = \log 0.0294/(0.0784 \cdot 0.1) = \log (3.75)
```

### Regenerate
The [`regenerate`](@ref) method takes a trace and generates an adjusted trace that is consistent with a change to the arguments to the function, and also generates new values for selected random choices.

**Example.**
Suppose we run [`regenerate`](@ref) on the example `trace`, with selection `:a` and `:b`:
```@example gfi_tutorial
(new_trace, w, _) = regenerate(trace, (), (), select(:a, :b))
```
Then, a new value for `:a` will be sampled from `bernoulli(0.3)`, and a new value for `:b` will be sampled from `bernoulli(0.4)`.
If the new value for `:b` is `true`, then the previous value for `:c` (`false`) will be retained.
If the new value for `:b` is `false`, then a new value for `:d` will be sampled from `bernoulli(0.7)`.
The previous value for `:c` will always be retained.
Suppose the new value for `:a` is `true`, and the new value for `:b` is `true`.
Then `get_choices(new_trace)` will be:
```julia
│
├── :a : true
│
├── :b : true
│
├── :c : false
│
└── :e : true
```
The weight (`w`) is ``\log 1 = 0``.


### Argdiffs

In addition to the input trace, and other arguments that indicate how to adjust the trace, each of these methods also accepts an **args** argument and an **argdiffs** argument, both of which are tuples.
The args argument contains the new arguments to the generative function, which may differ from the previous arguments to the generative function (which can be retrieved by applying [`get_args`](@ref) to the previous trace).
In many cases, the adjustment to the execution specified by the other arguments to these methods is 'small' and only affects certain parts of the computation.
Therefore, it is often possible to generate the new trace and the appropriate log probability ratios required for these methods without revisiting every state of the computation of the generative function.

To enable this, the argdiffs argument provides additional information about the *difference* between each of the previous arguments to the generative function, and its new argument value.
This argdiff information permits the implementation of the update method to avoid inspecting the entire argument data structure to identify which parts were updated.
Note that the correctness of the argdiff is in general not verified by Gen---passing incorrect argdiff information may result in incorrect behavior.

The trace update methods for all generative functions above should accept at least the following types of argdiffs:
- [`NoChange`](@ref)
- [`UnknownChange`](@ref)

Generative functions may also be able to process more specialized diff data types for each of their arguments, that allow more precise information about the different to be supplied.

### Retdiffs

To enable generative functions that invoke other functions to efficiently make use of incremental computation, the trace update methods of generative functions also return a **retdiff** value, which provides information about the difference in the return value of the previous trace an the return value of the new trace.

## Differentiable Programming

The trace of a generative function may support computation of gradients of its log probability with respect to some subset of (i) its arguments, (ii) values of random choice, and (iii) any of its **trainable parameters** (see below).

To compute gradients with respect to the arguments as well as certain selected random choices, use:

- [`choice_gradients`](@ref)

To compute gradients with respect to the arguments, and to increment a stateful gradient accumulator for the trainable parameters of the generative function, use:

- [`accumulate_param_gradients!`](@ref)

A generative function statically reports whether or not it is able to compute gradients with respect to each of its arguments, through the function [`has_argument_grads`](@ref).

### Trainable parameters

The **trainable parameters** of a generative function are (unlike arguments and random choices) *state* of the generative function itself, and are not contained in the trace.
Generative functions that have trainable parameters maintain *gradient accumulators* for these parameters, which get incremented by the gradient induced by the given trace by a call to [`accumulate_param_gradients!`](@ref).
Users then use these accumulated gradients to update to the values of the trainable parameters.

### Return value gradient

The set of elements (either arguments, random choices, or trainable parameters) for which gradients are available is called the **gradient source set**.
If the return value of the function is conditionally dependent on any element in the gradient source set given the arguments and values of all other random choices, for all possible traces of the function, then the generative function requires a *return value gradient* to compute gradients with respect to elements of the gradient source set.
This static property of the generative function is reported by [`accepts_output_grad`](@ref).

## API

The following GFI methods should be implemented for a [`Trace`](@ref):

```@docs
Trace
get_args
get_retval
get_choices
get_score
get_gen_fn
Base.getindex
```

The following GFI methods should be implemented for a [`GenerativeFunction`](@ref) and its associated [`Trace`](@ref) datatype:

```@docs
GenerativeFunction
simulate
generate
update
regenerate
project
propose
assess
```

Generative functions that support gradient computation with respect to arguments or trainable parameters should implement the following static properties:

```@docs
has_argument_grads
accepts_output_grad
choice_gradients
accumulate_param_gradients!
get_params
```
