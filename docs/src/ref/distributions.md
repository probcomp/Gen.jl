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

## Creating New Distributions via the Distributions DSL

In addition to using the above built-in distributions, and to defining subtypes
of `Gen.Distribution` in plain Julia, you may combine existing distributions to
create new ones using the `@dist` DSL.  The syntax for this DSL is

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
contain exactly one random choice.  The value of the `@dist` expression is then
a `Gen.distribution` object called `name`, parameterized by `arg1, ..., argN`,
representing the distribution over _return values_ of `body`.

This DSL is designed to address the issue that sometimes, values stored in the
trace do not correspond to the most natural physical elements of the model
state space, making inference programming and querying more taxing than
necessary. For example, suppose we have a model of classes at a school, where
the number of students is random, with mean 10, but always at least 3. Rather
than writing the model as

```julia
@gen function class_model()
   n_students = @trace(poisson(7), :n_students_minus_3) + 3
   ...
end
```

and thinking about the random variable `:n_students_minus_3`, you can use the
`@dist` DSL to instead write

```julia
@dist student_distr(mean, min) = poisson(mean-min) + min

@gen function class_model()
   n_students = @trace(student_distr(10, 3), :n_students)
   ...
end
```

and think about the more natural random variable `:n_students`.  This leads to
more natural inference programs, which can constrain and propose directly to
the `:n_students` trace address.

### Permitted means of combination

Of course, it is not possible for `@dist` to work on any arbitrary `body`.  We
now describe which constructs are permitted inside the `body` of a `@dist`
expression.

We can think of the `body` of an `@dist` function as containing ordinary Julia
code, except that in addition to being described by their ordinary Julia types,
each expression also belongs to one of three "type spaces." These are:

1. `CONST`: Constants, whose value is known at the time this `@dist` expression
   is evaluated.
2. `ARG`: Arguments and (deterministic, differentiable) functions of arguments.
   All expressions representing non-random values that depend on distribution
   arguments are `ARG` expressions.
3. `RND`: Random variables. All expressions whose runtime values may differ
   across multiple calls to this distribution (with the same arguments) are
   `RND` expressions.

**Importantly, Julia control flow constructs generally expect `CONST` values:
the condition of an `if` or the range of a `for` loop cannot be `ARG` or
`RND`.**
_(Developer note: We could change this: it would not be too difficult, e.g., to
add support for most use cases of `if`.)_

The body expression as a whole must be a `RND` expression, representing a
random variable. The behavior of the `@dist` definition is then to define a new
distribution (with name `name`) that samples and evaluates the logpdf of the
random variable represented by the `body` expression.

Expressions are typed compositionally, with the following typing rules:

1. **Literals and free variables are `CONST`s.** Literals and symbols that
   appear free in the `@dist` body are of type `CONST`.

2. **Arguments are `ARG`s.** Symbols bound as arguments in the `@dist`
   declaration have type `ARG` in its body.

3. **Drawing from a distribution gives `RND`.** If `d` is a distribution, and
   `x_i` are of type `ARG` or `CONST`, `d(x_1, x_2, ...)` is of type `RND`.

4. **Functions of `CONST`s are `CONST`s.** If `f` is a deterministic function
   and `x_i` are all of type `CONST`, `f(x_1, x_2, ...)` is of type `CONST`.

5. **Functions of `CONST`s and `ARG`s are `ARG`s.** If `f` is a
   _differentiable_ function, and each `x_i` is either a `CONST` or a _scalar_
   `ARG` (with at least one `x_i` being an `ARG`), then `f(x_1, x_2, ...)` is
   of type `ARG`.

6. **Functions of `CONST`s, `ARG`s, and `RND`s are `RND`s.** If `f` is one of a
   special set of deterministic functions we've defined (`+`, `-`, `*`, `/`,
   `exp`, `log`, `getindex`), and exactly one of its arguments `x_i` is of type
   `RND`, then `f(x_1, x_2, ...)` is of type `RND`.

One way to think about this, without all the rules, is that `CONST` values are
"contaminated" by interaction with `ARG` values (becoming `ARG`s themselves),
and both `CONST` and `ARG` are "contaminated" by interaction with `RND`.
Thinking of the body as an AST, the journey from leaf node to root node always
involves transitions in the direction of `CONST -> ARG -> RND`, never in
reverse. (There are certain similarities here with [security
types](https://en.wikipedia.org/wiki/Security_type_system), which also enforce
a single direction of information flow -- anything that touches classified data
is also classified. Also, Tabular v2's type system, which similarly separates
deterministic and random values in its type system.)

#### Restrictions
Users may _not_ reassign to arguments (like `x` in the above example), and may
not apply functions with side effects. Names bound to expressions of type `RND`
must be used only once. e.g., `let x = normal(0, 1) in x + x` is not allowed.

**TODO:** (Expand on scalar-vs.-vector problems here.)

#### Examples

Let's walk through some examples.

```julia
@dist f(x) = exp(normal(x, 1))
```

We can annotate with types:
```
1 :: CONST		  (by rule 1)
x :: ARG 		  (by rule 2)
normal(x, 1) :: RND 	  (by rule 3)
exp(normal(x, 1)) :: RND  (by rule 6)
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

Note that `getindex` is designed to work on anything indexible, not just
vectors. So, for example, it also works with Dicts.

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

## A note for the long-term future
There is a good amount of research on automatically computing densities of
probabilistic programs (considered as distributions over their return values),
for example [this very cool
paper](http://homes.sice.indiana.edu/ccshan/rational/disint2arg.pdf) by the
Hakaru people. They also rely on invertible (and piecewise invertible)
functions, change-of-variables, etc., but their approach is significantly more
sophisticated than what you see here (for one thing, it can handle many random
samples in the body of the probabilistic program).

Deriving a density calculator for an arbitrary probabilistic program is a sort
of "holy grail" (and in practice, it seems impossible to achieve -- this would
amount to computing any marginalization query).   It's nice because it frees
you (the querier and inference programmer) from reasoning about execution
traces; you can manually specify your state space (the return value) to be
parameterized however you want, and write proposals that operate directly on
that state space.

Adding this "Distribution DSL" to Gen — or really, more general versions that
incorporate the insights of these "density compilers" down the line — could
give Gen users the best of both worlds: for the parts of their computation that
we can automatically compute densities for, they can shove everything into one
`@trace` statement, and write their model programs and proposal programs as
distributions. For the rest, they can use generative functions.
