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
Generative functions behave like Julia functions in some respects.
For example, we can call a generative function `foo` on arguments and get an output value using regular Julia call syntax:
```julia-repl
>julia foo(2, 4)
7
```
However, generative functions are distinct from Julia functions because they support additional behaviors, described in the remainder of this section.

## Mathematical definition

Generative functions represent computations that accept some arguments, may use randomness internally, return an output, and cannot mutate externally observable state.
We represent the randomness used during an execution of a generative function as a map from unique **addresses** to values, denoted ``t : A \to V`` where ``A`` is an address set and ``V`` is a set of possible values that random choices can take.
In this section, we assume that random choices are discrete to simplify notation.
We say that two random choice maps ``t`` and ``s`` **agree** if they assign the same value for any address that is in both of their domains.

Generative functions may also use **non-addressed randomness**, which is not included in the map ``t``.
However, the state of non-addressed random choices *is* maintained by the trace internally.
We denote non-addressed randomness by ``r``.
Non-addressed randomness is useful for example, when calling black box Julia code that implements a randomized algorithm.

The observable behavior of every generative function is defined by the following mathematical objects:

### 1. Input type
The set of valid argument tuples to the function, denoted ``X``.

### 2. Probability distribution family
A family of probability distributions ``p(t, r; x)`` on maps ``t`` from random choice addresses to their values, and non-addressed randomness ``r``, indexed by arguments ``x``, for all ``x \in X``.
Note that the distribution must be normalized:
```math
\sum_{t, r} p(t, r; x) = 1 \;\; \mbox{for all} \;\; x \in X
```
This corresponds to a requirement that the function terminate with probabability 1 for all valid arguments.
We use ``p(t; x)`` to denote the marginal distribution on the map ``t``:
```math
p(t; x) := \sum_{r} p(t, r; x)
```
And we denote the conditional distribution on non-addressed randomness ``r``, given the map ``t``, as:
```math
p(r; x, t) := p(t, r; x) / p(t; x)
```

### 3. Return value function
A (deterministic) function ``f`` that maps the tuple ``(x, t)`` of the arguments and the random choice map to the return value of the function (which we denote by ``y``).
Note that the return value cannot depend on the non-addressed randomness.

### 4. Internal proposal distribution family
A family of probability distributions ``q(t; x, u)`` on maps ``t`` from random choice addresses to their values, indexed by tuples ``(x, u)`` where ``u`` is a map from random choice addresses to values, and where ``x`` are the arguments to the function.
It must satisfy the following conditions:
```math
\sum_{t} q(t; x, u) = 1 \;\; \mbox{for all} \;\; x \in X, u
```
```math
p(t; x) > 0 \mbox{ if and only if } q(t; x, u) > 0 \mbox{ for all } u \mbox{ where } u \mbox{ and } t \mbox{ agree }
```
```math
q(t; x, u) > 0 \mbox{ implies that } u \mbox{ and } t \mbox{ agree }.
```
There is also a family of probability distributions ``q(r; x, t)`` on non-addressed randomness, that satisfies:
```math
q(r; x, t) > 0 \mbox{ if and only if } p(r; x, t) > 0
```

## Traces

An **execution trace** (or just *trace*) is a record of an execution of a generative function.
There is no abstract type representing all traces.
Different concrete types of generative functions use different data structures and different Jula types for their traces.
The trace type that a generative function uses is the second type parameter of the [`GenerativeFunction`](@ref) abstract type.

A trace of a generative function can be produced using:
```@docs
initialize
```

The trace contains various information about the execution, including:

The arguments to the generative function:
```@docs
get_args
```

The return value of the generative function:
```@docs
get_retval
```

The map ``t`` from addresses of random choices to their values:
```@docs
get_assmt
```

The log probability that the random choices took the values they did:
```@docs
get_score
```

A reference to the generative function that was executed:
```@docs
get_gen_fn
```

## Trace update methods

It is often important to update or adjust the trace of a generative function.
In Gen, traces are **persistent data structures**, meaning they can be treated as immutable values.
There are several methods that take a trace of a generative function as input and return a new trace of the generative function based on adjustments to the execution history of the function.
We will illustrate these methods using the following generative function:
```julia
@gen function foo()
    val = @addr(bernoulli(0.3), :a)
    if @addr(bernoulli(0.4), :b)
        val = @addr(bernoulli(0.6), :c) && val
    else
        val = @addr(bernoulli(0.1), :d) && val
    end
    val = @addr(bernoulli(0.7), :e) && val
    return val
end
```
Suppose we have a trace (`trace`) with initial choices:
```
│
├── :a : false
│
├── :b : true
│
├── :c : false
│
└── :e : true
```
Note that address `:d` is not present because the branch in which `:d` is sampled was not taken because random choice `:b` had value `true`.

### Force Update
```@docs
force_update
```
Suppose we run [`force_update`](@ref) on the example `trace`, with the following constraints:
```
│
├── :b : false
│
└── :d : true
```
```julia
constraints = DynamicAssignment((:b, false), (:d, true))
(new_trace, w, discard, _) = force_update((), noargdiff, trace, constraints)
```
Then `get_assmt(new_trace)` will be:
```
│
├── :a : false
│
├── :b : false
│
├── :d : true
│
└── :e : true
```
and `discard` will be:
```
│
├── :b : true
│
└── :c : false
```
Note that the discard contains both the previous values of addresses that were overwritten, and the values for addresses that were in the previous trace but are no longer in the new trace.
The weight (`w`) is computed as:
```math
p(t'; x) = 0.7 × 0.4 × 0.4 × 0.7 = 0.0784\\
p(t; x') = 0.7 × 0.6 × 0.1 × 0.7 = 0.0294\\
w = \log p(t'; x')/p(t; x) = \log 0.0294/0.0784 = \log 0.375
```

### Free Update
```@docs
free_update
```
Suppose we run [`free_update`](@ref) on the example `trace`, with selection `:a` and `:b`:
```julia
(new_trace, w, _) = free_update((), noargdiff, trace, select(:a, :b))
```
Then, a new value for `:a` will be sampled from `bernoulli(0.3)`, and a new value for `:b` will be sampled from `bernoulli(0.4)`.
If the new value for `:b` is `true`, then the previous value for `:c` (`false`) will be retained.
If the new value for `:b` is `false`, then a new value for `:d` will be sampled from `bernoulli(0.7)`.
The previous value for `:c` will always be retained.
Suppose the new value for `:a` is `true`, and the new value for `:b` is `true`.
Then `get_assmt(new_trace)` will be:
```
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


### Extend
```@docs
extend
```

### Argdiffs

In addition to the input trace, and other arguments that indicate how to adjust the trace, each of these methods also accepts an **args** argument and an **argdiff** argument.
The args argument contains the new arguments to the generative function, which may differ from the previous arguments to the generative function (which can be retrieved by applying [`get_args`](@ref) to the previous trace).
In many cases, the adjustment to the execution specified by the other arguments to these methods is 'small' and only effects certain parts of the computation.
Therefore, it is often possible to generate the new trace and the appropriate log probability ratios required for these methods without revisiting every state of the computation of the generative function.
To enable this, the argdiff argument provides additional information about the *difference* between the previous arguments to the generative function, and the new arguments.
This argdiff information permits the implementation of the update method to avoid inspecting the entire argument data structure to identify which parts were updated.
Note that the correctness of the argdiff is in general not verified by Gen---passing incorrect argdiff information may result in incorrect behavior.

The trace update methods for all generative functions above should accept at least the following types of argdiffs:
```@docs
NoArgDiff
UnknownArgDiff
```
Generative functions may also accept custom types for their argdiffs that allow more precise information about the different to be supplied.
It is the responsibility of the author of a generative function to specify the valid argdiff types in the documentation of their function, and it is the responsibility of the user of a generative function to construct and pass in the appropriate argdiff value.

### Retdiffs

To enable generative functions that invoke other functions to efficiently make use of incremental computation, the trace update methods of generative functions also return a **retdiff** value, which provides information about the difference in the return value of the previous trace an the return value of the new trace.

Generative functions may return arbitrary retdiff values, provided that the type has the following method:
```@docs
isnodiff
```
It is the responsibility of the author of the generative function to document the possible retdiff values that may be returned, and how the should be interpreted.
There are two generic constant retdiff provided for authors of generative functions to use in simple cases:
```@docs
DefaultRetDiff
NoRetDiff
```

## Differentiable programming

Generative functions may support computation of gradients with respect to (i) all or a subset of its arguments, (ii) its **trainable parameters**, and (iii) the value of certain random choices.
The set of elements (either arguments, trainable parameters, or random choices) for which gradients are available is called the *gradient source set*.
A generative function statically reports whether or not it is able to compute gradients with respect to each of its arguments, through the function `has_argument_grads`.
Let ``x_G`` denote the set of arguments for which the generative function does support gradient computation.
Similarly, a generative function supports gradients with respect the value of random choices made at all or a subset of addresses.
If the return value of the function is conditionally independent of each element in the gradient source set given the other elements in the gradient source set and values of all other random choices, for all possible traces of the function, then the generative function requires a *return value gradient* to compute gradients with respect to elements of the gradient source set.
This static property of the generative function is reported by `accepts_output_grad`.

```@docs
has_argument_grads
accepts_output_grad
backprop_params
backprop_trace
get_params
```

### Additional methods
```@docs
project
propose
assess
```

## Custom generative function types

Most users can just use generative functions written in the [Built-in Modeling Language](@ref), and can skip this section.
However, to develop new modeling DSLs, or optimized implementations of certain probabilistic modeling components, users can also implement custom types of generative functions.
We recommend the following steps for implementing a new type of generative function, and also looking at the implementation for the [`DynamicDSLFunction`](@ref) type as an example.

### Define a trace data type
```julia
struct MyTraceType
    ..
end
```

### Decide the return type for the generative function
Suppose our return type is `Vector{Float64}`.

### Define a data type for your generative function
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

### Decide what the arguments to a generative function should be
For example, our generative functions might take two arguments, `a` (of type `Int`) and `b` (of type `Float64`).
Then, the argument tuple passed to e.g. [`initialize`](@ref) will have two elements.

NOTE: Be careful to distinguish between arguments to the generative function itself, and arguments to the constructor of the generative function.
For example, if you have a generative function type that is parametrized by, for example, modeling DSL code, this DSL code would be a parameter of the generative function constructor.

### Decide what the addressed random choices (if any) will be
Remember that each random choice is assigned a unique address in (possibly) hierarchical address space.
You are free to design this address space as you wish, although you should document it for users of your generative function type.

### Implement the methods of the interface

- At minimum, you need to implement all methods under the [`Traces`](@ref) heading (e.g. [`initialize`](@ref), ..)

- To support [`metropolis_hastings`](@ref) or local optimization, or local iterative adjustments to traces, be sure to implement the [`force_update`](@ref) and [`free_update](@ref) methods.

- To support gradients of the log probability density with respect to the arguments and/or random choices made by the function, implement the [`backprop_trace`](@ref) method.

- Generative functions can also have trainable parameters (e.g. neural network weights). To support these, implement the [`backprop_params`](@ref) method.

- To support use of your generative function in custom proposals (instead of just generative models), implement [`assess`](@ref) and [`propose`](@ref) methods.

