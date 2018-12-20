# Modeling Languages

Gen provides two embedded DSLs for defining generative functions.
Both DSLs are based on Julia function definition syntax.

## Dynamic DSL

*Dynamic DSL* functions are defined using the `@gen` macro in front of a Julia function definition.
We will refer to Dynamic DSL functions as `@gen` functions from here forward.

Here is a sample `@gen` function that samples two random choices:
```julia
@gen function foo(prob::Float64)
    z1 = @addr(bernoulli(prob), :a)
    z2 = @addr(bernoulli(prob), :b)
    return z1 || z2
end
```

After running this code, `foo` is a Julia value of type `DynamicDSLFunction`, which is a subtype of `GenerativeFunction`.

### Making Random Choices

Random choices are made by applying a [Probability Distribution](@ref) to some arguments:
```julia
val::Bool = bernoulli(0.5)
```
Within the context of a `@gen` function, wrapping the right-hand side of the expression associates the random choice with an *address*, and evaluates to the value of the random choice.
Addresses can be any Julia value.
Here, we give the Julia symbol address `:z`:
```julia
val::Bool = @addr(bernoulli(0.5), :z)
```
Not all random choices need to be given addresses.
An address is required if the random choice will be observed, or will be referenced by a custom inference algorithm (e.g. if it will be proposed to by a custom proposal distribution).


### Calling Generative Functions

`@gen` functions can invoke other generative functions in three possible ways.

**Untraced call**:
If `foo` is a generative function, we can invoke `foo` from within the body of a `@gen` function using regular call syntax.
The random choices made within the call are not given addresses in our trace, and are therefore *non-addressable* random choices (see [Generative Function Interface](@ref) for details on non-addressable random choices).
```julia
val = foo(0.5)
```

**Traced call with shared namespace**:
We can include the addressable random choices made by `foo` in the caller's trace using `@splice`:
```julia
val = @splice(foo(0.5))
```
Now, all random choices made by `foo` are included in our trace.
The caller must guarantee that there are no address collisions.

**Traced call with hierarchical namespace**:
We can include the addressable random choices made by `foo` in the caller's trace, under a namespace, using `@addr`:
```julia
val = @addr(foo(0.5), :x)
```
Now, all random choices made by `foo` are included in our trace, under the namespace `:x`.
For example, if `foo` makes random choices at addresses `:a` and `:b`, these choices will have addresses `:x => :a` and `:x => :b` in the caller's trace.

### Using Composite Addresses

In Julia, `Pair` values can be constructed using the `=>` operator.
For example, `:a => :b` is equivalent to `Pair(:a, :b)` and `:a => :b => :c` is equivalent to `Pair(:a, Pair(:b, :c))`.
A `Pair` value (e.g. `:a => :b => :c`) can be passed as the address field in an `@addr` expression, provided that there is not also a random choice or generative function called with `@addr` at any prefix of the address.

Concretely, the following are *invalid* within the body of a generative function:
```julia
@addr(normal(0, 1), :a => :b => :c)
@addr(normal(0, 1), :a => :b)
```
```julia
@addr(normal(0, 1), :a => :b => :c)
@addr(normal(0, 1), :a)
```
```julia
@addr(normal(0, 1), :a => :b => :c)
@addr(foo(0.5), :a => :b)
```
```julia
@addr(normal(0, 1), :a)
@addr(foo(0.5), :a => :b)
```
The following are *valid*:
```julia
@addr(normal(0, 1), :a => :b)
@addr(normal(0, 1), :a => :c)
```
```julia
@addr(normal(0, 1), :a => :b)
@addr(foo(0.5), :a => :c)
```


### Returning

Like regular Julia functions, `@gen` functions return either the expression used in a `return` keyword, or by evaluating the last expression in the function body.
Note that the return value of a `@gen` function is distinct from its trace.


## Static DSL

*Static DSL* functions are defined using the `@staticgen` macro.
