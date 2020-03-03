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
laplace
```

# Description of Distribution DSL

## The DSL: A user's perspective

Users may now define new distributions using the `@dist` DSL.

Its syntax is

```julia
@dist name(arg1, arg2, ..., argN) = body
```
or
```julia
@dist function name(arg1, arg2, ..., argN)
    body
end
```

The idea is for the user to write ordinary Julia code in the `body`, which makes and transforms a single random choice; in many cases, this should "just work", creating a Gen distribution object called `name`, parameterized by `arg1, ..., argN`, representing the distribution over _return values_ of the body.

This is designed to address the issue that sometimes, values stored in the trace do not correspond to the most natural physical elements of the model state space, making inference programming and querying more taxing than necessary. For example, suppose we have a model of classes at a school, where the number of students is random, with mean 10, but always at least 3. In Gen today, you might write the model as follows:

```julia
@gen function class_model()
   n_students = @trace(poisson(7), :n_students_minus_3) + 3
   ...
end
```

Then in all your queries and proposals, you'd have to think about `:n_students_minus_3` instead of the more natural `:n_students`. With the `@dist` DSL, you can write:

```julia
@dist student_distr(mean, min) = poisson(mean-min) + min

@gen function class_model()
   n_students = @trace(student_distr(10, 3), :n_students)
   ...
end
```

This leads to more natural inference programs, which can constrain and propose directly to the `:n_students` trace address.

Of course, it is not possible to implement this feature for any possible`@dist` declaration. So to use the DSL safely for more complicated tasks, it is useful to have a mental model for what is permitted. (Otherwise, the user may see static errors when evaluating the `@dist` definition.)

We can think of the `body` of an `@dist` function as containing ordinary Julia code, except that in addition to being described by their ordinary Julia types, each expression also belongs to one of three "type spaces." These are:

1. `CONST`: Constants, whose value is known at the time this `@dist` expression is evaluated.
2. `ARG`: Arguments and (deterministic, differentiable) functions of arguments. All expressions representing non-random values that depend on distribution arguments are `ARG` expressions.
3. `RND`: Random variables. All expressions whose runtime values may differ across multiple calls to this distribution (with the same arguments) are `RND` expressions.

**Importantly, Julia control flow constructs generally expect `CONST` values: the condition of an `if` or the range of a `for` loop cannot be `ARG` or `RND`. (We could change this: it would not be too difficult, e.g., to add support for most use cases of `if`.)**

The body expression as a whole must be a `RND` expression, representing a random variable. The behavior of the `@dist` definition is then to define a new distribution (with name `name`) that samples and evaluates the logpdf of the random variable represented by the `body` expression.

Expressions are typed compositionally, with the following typing rules:

1. **Literals and free variables are `CONST`s.** Literals and symbols that appear free in the `@dist` body are of type `CONST`.
2. **Arguments are `ARG`s.** Symbols bound as arguments in the `@dist` declaration have type `ARG` in its body.
3. **Drawing from a distribution gives `RND`.** If `d` is a distribution, and `x_i` are of type `ARG` or `CONST`, `d(x_1, x_2, ...)` is of type `RND`.
4. **Functions of `CONST`s are `CONST`s.** If `f` is a deterministic function and `x_i` are all of type `CONST`, `f(x_1, x_2, ...)` is of type `CONST`.
5. **Functions of `CONST`s and `ARG`s are `ARG`s.** If `f` is a _differentiable_ function, and each `x_i` is either a `CONST` or a _scalar_ `ARG` (with at least one `x_i` being an `ARG`), then `f(x_1, x_2, ...)` is of type `ARG`.
6. **Functions of `CONST`s, `ARG`s, and `RND`s are `RND`s.** If `f` is one of a special set of deterministic functions we've defined (`+`, `-`, `*`, `/`, `exp`, `log`, `getindex`), and exactly one of its arguments `x_i` is of type `RND`, then `f(x_1, x_2, ...)` is of type `RND`.

One way to think about this, without all the rules, is that `CONST` values are "contaminated" by interaction with `ARG` values (becoming `ARG`s themselves), and both `CONST` and `ARG` are "contaminated" by interaction with `RND`. Thinking of the body as an AST, the journey from leaf node to root node always involves transitions in the direction of `CONST -> ARG -> RND`, never in reverse. (There are certain similarities here with [security types](https://en.wikipedia.org/wiki/Security_type_system), which also enforce a single direction of information flow -- anything that touches classified data is also classified. Also, Tabular v2's type system, which similarly separates deterministic and random values in its type system.)

### Examples

Let's walk through some examples.

```
@dist f(x) = exp(normal(x, 1))
```

We can annotate with types:
```
1 :: CONST		 (by rule 1)
x :: ARG 		 (by rule 2)
normal(x, 1) :: RND 	 (by rule 3)
exp(normal(x, 1)) :: RND (by rule 6)
```

Here's another:
```
@dist function labeled_cat(labels, probs)
	index = categorical(probs)
	labels[index]
end
```

And the types:
```
probs :: ARG 			(by rule 2)
categorical(probs) :: RND 	(by rule 3)
index :: RND 			(Julia assignment)
labels :: ARG 			(by rule 2)
labels[index] :: RND 		(by rule 6, f == getindex)
```

Note that `getindex` is designed to work on anything indexible, not just vectors. So, for example, it also works with Dicts.

Another one (not as realistic, but it uses all the rules):

```
@dist function weird(x)
  log(normal(exp(x), exp(x))) + (x * (2 + 3))
end
```

And the types:

```
2, 3 :: CONST 						(by rule 1)
2 + 3 :: CONST 						(by rule 4)
x :: ARG 						(by rule 2)
x * (2 + 3) :: ARG 					(by rule 5)
exp(x) :: ARG 						(by rule 5)
normal(exp(x), exp(x)) :: RND 				(by rule 3)
log(normal(exp(x), exp(x))) :: RND 			(by rule 6)
log(normal(exp(x), exp(x))) + (x * (2 + 3)) :: RND 	(by rule 6)
```

### Restrictions
Users may _not_ reassign to arguments (like `x` in the above example), and may not apply functions with side effects. Names bound to expressions of type RND must be used only once. e.g., `let x = normal(0, 1) in x + x` is not allowed. (Thoughts on lifting this restriction appear further below.)

**TODO:** (Expand on scalar-vs.-vector problems here.)


## Implementation
The `@dist` macro transforms an expression of the form

```
@dist f(x, y, z) = body...
```

into an expression of the form

```
f = compile_dist_with_args(transformed_body..., n_args)
```

Within `transformed_body`, values of type `ARG` are represented as Julia values of type `Arg`; values of type `RND` are represented as Julia values of type `DistWithArgs`; and values of type `CONST` are just ordinary Julia values. The syntactic transformations that achieve this are:

1. All occurrences of the symbols used for argument names (`x`, `y`, and `z` above) are replaced with `SimpleArg(i)`, where `i` is the positional index of the argument. For example, `x` is replaced with `SimpleArg(1)` everywhere it appears, whereas `z` is `SimpleArg(3)`.
2. All function calls `f(...)` are replaced with `dist_call(f, ...)`. This allows new semantics to be given to all function calls, as described below.

Most of the typing rules are handled by `dist_call`:

1. If the function being called is a distribution, `dist_call` returns a `DistWithArgs` instead of sampling from the distribution. As mentioned above, the `DistWithArgs` type represents an expression of type `RND`, i.e., an expression representing a random variable. The details of its representation are described further below. (_Note: a possible extension that occurs to me now is that each `DistWithArgs` created in this way could be given a unique _tag_, which persisted as the variable was transformed. This way, we could distinguish between `let x = normal(0, 1) in x + x` and `normal(0, 1) + normal(0, 1)`: in the first case, the two distributions have the same tag, whereas in the second, they do not. We could then implement `+`, e.g., for distributions with the same tag._)

2. If any of the arguments to the function being called are of type `DistWithArgs` (i.e., are of type `RND`), we simply call the function as normal. This relies on there being a method of `f` defined on arguments of type `DistWithArgs`, that returns a modified `DistWithArgs`. We have implemented such methods for `+`, `-`, `*`, `/`, `exp`, `log`, and `getindex`.

3. If all arguments are deterministic but at least one is of type `Arg`, we do not call the function yet, instead producing a `TransformedArg <: Arg`. `TransformedArg`s represent functions of arguments, and are described further below.

4. If all arguments are `CONST` (i.e., none are `Arg` or `DistWithArgs` values), we simply apply the function as usual.


Note that `dist_call` is allowed to be slow, as it runs only at compile-time.


The combination of `@dist` and `dist_call` implement the six typing rules described in the first section: rule 1 is satisfied trivially, rule 2 is enforced by `@dist`'s sweep over symbols in the body, and rules 3, 4, 5, and 6 correspond to the four cases of `dist_call` above.

There remain four issues to discuss:
1. How are expressions of type `ARG` (especially `TransformedArg`) represented concretely?
2. How are expressions of type `RND` (i.e., `DistWithArgs`) represented concretely?
3. How are the special functions `+`, `-`, `*`, `/`, `exp`, `log`, and `getindex`, which accept and return values of type `DistWithArgs`, implemented?
4. What does the `compile_dist_with_args` step do?

## Representations of `ARG`-type expressions

We use an abstract type in Julia, `Arg`, for expressions with "type space" `ARG`. Under the abstract Julia `Arg` type, there are two concrete types that can represent such expressions:

1. A `SimpleArg` represents a direct use, within the `@dist` body, of one of the arguments to the `@dist` function. It stores an `Int8` representing its position in the distribution's argument list. 

2. A `TransformedArg` represents a function of (potentially multiple) `@dist` arguments and constants. It stores (a) the function `orig_f` being applied, (b) a list `f_args` of other `Arg` values that appear as arguments to the function `orig_f`, and (c) an "arg passer," a function that accepts `orig_f` and concrete values for arguments in `f_args`, and applies `orig_f` to those arguments and any relevant constant values. For example, if `x` and `y` are of type `Arg`, then `g(x, 1, y)` would be represented as a `TransformedArg` with `orig_f == g`, `f_args == [x, y]`, and `arg_passer == (x, y) -> g(x, 1, y)`.

Values of type `Arg` support two Julia methods: `eval_arg(x, actual_args)`, which takes in an `Arg` `x` and a vector of concrete arguments to this `@dist`, and returns `x`'s concrete value; and `all_indices`, which returns a list of positions (in the original `@dist` arg list) that are used in computing this arg's value.

## Representations of `RND`-type expressions

A random variable (`RND`) is represented as a Julia `DistWithArgs`, which, as its name suggests, contains two things: a base distribution, and a list of arguments to the distribution. The list of arguments may contain both constants (i.e., ordinary Julia values) and values of type `Arg`. One way to think of it is that a `DistWithArgs` is a partially applied distribution function, but which arguments are "unapplied" (which are type `Arg` rather than constants) is flexible.

A `DistWithArgs` is not itself a Gen distribution, but `compile_dist_with_args` turns it into one (see two sections down).


## Implementing functions of distributions
In order to implement functions like `+` and `exp` and `getindex`, we introduce three new "distribution combinators":

1. A `TransformedDistribution` represents an arbitrary invertible, twice-differentiable function applied to a base distribution. (If the base distribution is discrete, the differentiability requirement is nullified.) The invertible function must be invertible in its first argument, which we take to be a value sampled from the base distribution. The arguments to the TransformedDistribution are a concatenation of (a) the second-and-beyond arguments to the invertible transformation, and (b) the original arguments of the base distribution.

2. A `RelabeledDistribution` consists of a base distribution `base` over values of some index type, and a collection `coll` indexed by that type. Samples from the RelabeledDistribution are generated by drawing an index `i` from `base`, then looking up `coll[i]`. The arguments to this distribution are the same as to the base distribution.

3. A `WithLabelArg` distribution stores only a base distribution, but adds an additional argument to the front: a collection indexed by the values sampled from the base distribution. It works just like `RelabeledDistribution`, except that instead of storing the collection, it accepts the collection as an additional argument.

Given these combinators, we can implement, for example, the following functions on `DistWithArgs` values:

```julia
# Addition
Base.:+(b::DistWithArgs{T}, a::Real) where T <: Real = DistWithArgs(TransformedDistribution{T, T}(b.base, 0, x -> x + a, x -> x - a, x -> (1.0,)), b.arglist)
Base.:+(b::DistWithArgs{T}, a::Arg) where T <: Real = DistWithArgs(TransformedDistribution{T, T}(b.base, 1, +, -, (x, a) -> (1.0, -1.0)), (a, b.arglist...))
Base.:+(a::Real, b::DistWithArgs{T}) where T <: Real = b + a
Base.:+(a::Arg, b::DistWithArgs{T}) where T <: Real = b + a

# Exponentiation
Base.exp(b::DistWithArgs{T}) where T <: Real = DistWithArgs(TransformedDistribution{T, T}(b.base, 0, exp, log, x -> (1.0 / x,)), b.arglist)
Base.log(b::DistWithArgs{T}) where T <: Real = DistWithArgs(TransformedDistribution{T, T}(b.base, 0, log, exp, x -> (exp(x),)), b.arglist)

# Indexing
Base.getindex(collection::Arg, d::DistWithArgs{T}) where T = DistWithArgs{Any}(WithLabelArg{Any, T}(d.base), (collection, d.arglist...))
Base.getindex(collection::AbstractArray{T}, d::DistWithArgs{U}) where {T, U} = DistWithArgs{T}(RelabeledDistribution{T, U}(d.base, collection), d.arglist)
Base.getindex(collection::AbstractDict{T}, d::DistWithArgs{U}) where {T, U} = DistWithArgs{T}(RelabeledDistribution{T, U}(d.base, collection), d.arglist)
```
Note how these functions manipulate both the base distribution and the argument list as necessary, to create the desired "partially applied" distribution.

## Creating a final distribution
The final step of evaluating an `@dist` expression is the call to `compile_dist_with_args`, which creates a `Distribution` out of a `DistWithArgs`. That distribution is of type `CompiledDistWithArgs`, which is essentially a `DistWithArgs`, but (a) with the total number of arguments `n_args` stored in the struct, and (b) a stored, precomputed answer to `has_argument_grads`.

Note that `n_args` is not simply the length of the `arglist`. A simple way to see this is to consider a distribution like the following:

```
@dist f(x, y, z, w) = normal(x + y * w, 1)
```

Here, the `CompiledDistWithArgs` will have `n_args == 4`, to represent the number of arguments that it accepts. However, its base distribution will be `normal` and its `arglist` will contain only two entries, representing the arguments to the `normal` distribution! The first entry will be a `TransformedArg` representing `x + y * w`, and the second will be the constant `1`.

Keeping straight these two notions of "argument" is probably the trickiest part of reading the source code.

## A note for the long-term future
There is a good amount of research on automatically computing densities of probabilistic programs (considered as distributions over their return values), most recently [this very cool paper](http://homes.sice.indiana.edu/ccshan/rational/disint2arg.pdf) by the Hakaru people. They also rely on invertible (and piecewise invertible) functions, change-of-variables, etc., but their approach is significantly more sophisticated than what you see here (for one thing, it can handle many random samples in the body of the probabilistic program).

Deriving a density calculator for an arbitrary probabilistic program is a sort of "holy grail" (and in practice, it seems impossible to achieve -- this would amount to computing any marginalization query).   It's nice because it frees you (the querier and inference programmer) from reasoning about execution traces; you can manually specify your state space (the return value) to be parameterized however you want, and write proposals that operate directly on that state space.

Adding this "Distribution DSL" to Gen — or really, more general versions that incorporate the insights of these "density compilers" down the line — could give Gen users the best of both worlds: for the parts of their computation that we can automatically compute densities for, they can shove everything into one `@trace` statement, and write their model programs and proposal programs as distributions. For the rest, they can use generative functions.
