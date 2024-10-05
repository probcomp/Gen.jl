# [Static Modeling Language](@id sml)

The **Static Modeling Language (SML)** is a restricted variant of the built-in [dynamic modeling language](@ref dml).
Models written in the static modeling language can result in better inference performance (more inference operations per second and less memory consumption), than the full built-in modeling language, especially for models used with iterative inference algorithms like [Markov Chain Monte Carlo](@ref mcmc) or [particle filtering](@ref particle_filtering).

A function is identified as using the static modeling language by adding the `static` annotation to the function.
For example:
```julia
@gen (static) function foo(prob::Float64)
    z1 = @trace(bernoulli(prob), :a)
    z2 = @trace(bernoulli(prob), :b)
    z3 = z1 || z2
    z4 = !z3
    return z4
end
```
After running this code, `foo` is a Julia value whose type is a subtype of `StaticIRGenerativeFunction`, which is a subtype of [`GenerativeFunction`](@ref).

## Static Computation Graphs
Using the `static` annotation instructs Gen to statically construct a directed acyclic graph for the computation represented by the body of the function.
For the function `foo` above, the static graph looks like:
```@raw html
<div style="text-align:center">
    <img src="../../../assets/static_graph.png" alt="example static computation graph" width="50%"/>
</div>
```
In this graph, oval nodes represent random choices, square nodes represent Julia computations, and diamond nodes represent arguments.
The light blue shaded node is the return value of the function.
Having access to the static graph allows Gen to generate specialized code for [Updating traces](@ref) that skips unecessary parts of the computation.
Specifically, when applying an update operation, the graph is analyzed, and each value in the graph identified as having possibly changed, or not.
Nodes in the graph do not need to be re-executed if none of their input values could have possibly changed.
Also, even if some inputs to a generative function node may have changed, knowledge that some of the inputs have not changed often allows the generative function being called to more efficiently perform its update operation.
This is the case for functions produced by [Generative Function Combinators](@ref combinators).

You can plot the graph for a function with the `static` annotation if you have PyCall installed, and a Python environment that contains the [graphviz](https://pypi.org/project/graphviz/) Python package, using, e.g.:
```julia
using PyCall
@pyimport graphviz
using Gen: draw_graph
draw_graph(foo, graphviz, "test")
```
This will produce a file `test.pdf` in the current working directory containing the rendered graph.

## Restrictions

First, the definition of a `(static)` generative function is always expected to occur as a [top-level definition](https://docs.julialang.org/en/v1/manual/modules/) (aka global variable); usage in non–top-level scopes is unsupported and may result in incorrect behavior.

Next, in order to be able to construct the static graph, Gen restricts the permitted syntax that can be used in functions annotated with `static`.
In particular, each statement in the body must be one of the following:

- A `@param` statement specifying any [trainable parameters](@ref trainable_parameters_modeling), e.g.:

```julia
@param theta::Float64
```

- An assignment, with a symbol or tuple of symbols on the left-hand side, and a Julia expression on the right-hand side, which may include `@trace` expressions, e.g.:

```julia
mu, sigma = @trace(bernoulli(p), :x) ? (mu1, sigma1) : (mu2, sigma2)
```

- A top-level `@trace` expression, e.g.:

```julia
@trace(bernoulli(1-prob_tails), :flip)
```
All `@trace` expressions must use a literal Julia symbol for the first component in the address. Unlike the full built-in modeling-language, the address is not optional.

- A `return` statement, with a Julia expression on the right-hand side, e.g.:

```julia
return @trace(geometric(prob), :n_flips) + 1
```

The functions are also subject to the following restrictions:

- Default argument values are not supported.

- Constructing named or anonymous Julia functions (and closures) is not allowed.

- List comprehensions with internal `@trace` calls are not allowed.

- Splatting within `@trace` calls is not supported

- Generative functions that are passed in as arguments cannot be traced.

- For composite addresses (e.g. `:a => 2 => :c`) the first component of the address must be a literal symbol, and there may only be one statement in the function body that uses this symbol for the first component of its address.

- Julia control flow constructs (e.g. `if`, `for`, `while`) cannot be used as top-level statements in the function body. Control flow should be implemented inside either Julia functions that are called, or other generative functions.

Certain loop constructs can be implemented using [Generative Function Combinators](@ref combinators) instead. For example, the following loop:
```julia
for (i, prob) in enumerate(probs)
    @trace(foo(prob), :foo => i)
end
```
can instead be implemented as:
```julia
@trace(Map(foo)(probs), :foo)
```

## Loading generated functions

Once a `(static)` generative function is defined, it can be used in the same way as a non-static generative function. In previous versions of Gen, the `@load_generated_functions` macro had to be called before a function with a `(static)` annotation could be used. This macro is no longer necessary, and will be removed in a future release.

## Performance tips
For better performance when the arguments are simple data types like `Float64`, annotate the arguments with the concrete type.
This permits a more optimized trace data structure to be generated for the generative function.

## Caching Julia values

By default, the values of Julia computations (all calls that are not random choices or calls to generative functions) are cached as part of the trace, so that [Updating traces](@ref) can avoid unecessary re-execution of Julia code.
However, this cache may grow the memory footprint of a trace.
To disable caching of Julia values, use the function annotation `nojuliacache` (this annotation is ignored unless the `static` function annotation is also used).
