# Gen Introduction

Gen is an extensible and reasonably performant probabilistic computing platform that makes it easier to develop probabilistic inference and learning applications.

## Generative Functions, Traces, and Assignments

Stochastic generative processes are represented in Gen as *generative functions*.
Generative functions are Julia functions that have been annotated using the `@gen` macro.
The generative function below takes a vector of x-coordinates, randomly generates the slope and intercept parameters of a line, and returns a random vector of y-coordinates sampled near that line at the given x-coordinates:

```julia
@gen function regression_model(xs::Vector{Float64})
    slope = normal(0, 2)
    intercept = normal(0, 2)
    ys = Vector{Float64}(undef, length(xs))
    for (i, x) in enumerate(xs)
        ys[i] = normal(slope * xs + intercept, 1.)
    end
    return ys
end
```

We can evaluate the generative function:

```julia
ys = regression_model([-5.0, -3.0, 0.0, 3.0, 5.0])
```

Above we have used a generative function to implement a simulation.
However, what distinguishes generative functions from plain-old simulators is their ability to be *traced*.
When we *trace* a generative function, we record the random choices that it makes, as well as additional data about the evaluation of the function.
This capability makes it possible to implement algorithms for probabilistic inference.
To trace a random choice, we need to give it a unique *address*, using the `@addr` keyword.
Here we give addresses to each of the random choices:

```julia
@gen function regression_model(xs::Vector{Float64})
    slope = @addr(normal(0, 2), "slope")
    intercept = @addr(normal(0, 2), "intercept")
    ys = Vector{Float64}(undef, length(xs))
    for (i, x) in enumerate(xs)
        ys[i] = @addr(normal(slope * xs[i] + intercept, 1.), "y-$i")
    end
    return ys
end
```

Addresses can be arbitrary Julia values except for `Pair`.
Julia symbols, strings, and integers are common types to use for addresses.

We trace a generative function using the `simulate` method.
We provide the arguments to the function in a tuple:

```julia
xs = [-5.0, -3.0, 0.0, 3.0]
trace = simulate(regression_model, (xs,))
```

The trace that is returned is a form of stack trace of the generative function that contains, among other things, the values for the random choices that were annotated with `@addr`.
To extract the values of the random choices from a trace, we use the method `get_assignment`:

```julia
assignment = get_assignment(trace)
```

The `get_assignment` method returns an *assignment*, which is a trie (prefix tree) that contains the values of random choices.
Printing the assignment gives a pretty-printed representation:

```julia
print(assignment)
```

```
│
├── "y-2" : -1.7800101128038626
│
├── "y-3" : 0.1832573462320619
│
├── "intercept" : 1.7434641799887896
│
├── "y-4" : 5.074512278024528
│
├── "slope" : 1.5232349541190595
│
└── "y-1" : -4.978881121779669
```

Generative functions can also call other generative functions; these calls can also be traced using `@addr`:

```julia
@gen function generate_params()
    slope = @addr(normal(0, 2), "slope")
    intercept = @addr(normal(0, 2), "intercept")
    return (slope, intercept)
end

@gen function generate_datum(x, slope, intercept)
    return @addr(normal(slope * x + intercept, 1.), "y")
end

@gen function regression_model(xs::Vector{Float64})
    (slope, intercept) = @addr(generate_params(), "parameters")
    ys = Vector{Float64}(undef, length(xs))
    for (i, x) in enumerate(xs)
        ys[i] = @addr(generate_datum(xs[i], slope, intercept), i)
    end
    return ys
end
```

This results in a hierarchical assignment:

```julia
trace = simulate(regression_model, (xs,))
assignment = get_assignment(trace)
print(assignment)
```

```
│
├── 2
│   │
│   └── "y" : -3.264252749715529
│
├── 3
│   │
│   └── "y" : -2.3036480286819865
│
├── "parameters"
│   │
│   ├── "intercept" : -0.8767368668034233
│   │
│   └── "slope" : 0.9082675922758383
│
├── 4
│   │
│   └── "y" : 2.4971551239517695
│
└── 1
    │
    └── "y" : -7.561723378403817
```

We can read values from a assignment using the following syntax:
```julia
assignment["intercept"]
```

To read the value at a hierarchical address, we provide a `Pair` where the first element of the pair is the first part ofthe hierarchical address, and the second element is the rest of the address.
For example:
```julia
assignment[1 => "y"]
```

Julia provides the operator `=>` for constructing `Pair` values.
Long hierarchical addresses can be constructed by chaining this operator, which associates right:
```julia
assignment[1 => "y" => :foo => :bar]
```

Generative functions can also write to hierarchical addresses directly:

```julia
@gen function regression_model(xs::Vector{Float64})
    slope = @addr(normal(0, 2), "slope")
    intercept = @addr(normal(0, 2), "intercept")
    ys = Vector{Float64}(undef, length(xs))
    for (i, x) in enumerate(xs)
        ys[i] = @addr(normal(slope * xs[i] + intercept, 1.), i => "y")
    end
    return ys
end
```

```julia
trace = simulate(regression_model, (xs,))
assignment = get_assignment(trace)
print(assignment)
```

```
│
├── "intercept" : -1.340778590777462
│
├── "slope" : -2.0846094796654686
│
├── 2
│   │
│   └── "y" : 3.64234023192473
│
├── 3
│   │
│   └── "y" : -1.5439406188116667
│
├── 4
│   │
│   └── "y" : -8.655741483764384
│
└── 1
    │
    └── "y" : 9.451320138931484
```

## Implementing Inference Algorithms

## Implementing Gradient-Based Learning

## Probabilistic Modules

## Compiled Generative Functions

## Incremental Computation

Getting good asymptotic scaling for iterative local search algorithms like MCMC or MAP optimization relies on the ability to update a trace efficiently, when a small number of random choice(s) are changed, or when there is a small change to the arguments of the function.

To support this, generative functions use *argdiffs* and *retdiffs*, which describe the change made to the arguments of the generative function, relative to the arguments in the previous trace, and the change to the return value of a generative function, relative to the return value in the previous trace.
The update generative function API methods `update`, `fix_update`, and `extend` accept the argdiff value, alongside the new arguments to the function, the previous trace, and other parameters; and return the new trace and the retdiff value.

### Argdiffs

An argument difference value, or *argdiff*, is associated with a pair of argument tuples `args::Tuple` and `new_args::Tuple`.
The update methods for a generative function accept different types of argdiff values, that depend on the generative function.
Two singleton data types are provided for expressing that there is no difference to the argument (`noargdiff::NoArgDiff`) and that there is an unknown difference in the arguments (`unknownargdiff::UnknownArgDiff`).
Generative functions may or may not accept these types as argdiffs, depending on the generative function.

### Retdiffs

A return value difference value, or *retdiff*, is associated with a pair of traces.
The update methods for a generative function return different types of retdiff values, depending on the generative function.
The only requirement placed on retdiff values is that they implement the `isnodiff` method, which takes a retdiff value and returns `true` if there was no change in the return value, and otherwise returns false.

### Custom incremental computation in embedded modeling DSL

For generative functions expressed in the embedded modeling DSL, retdiff values are computed by user *diff code* that is placed inline in the body of the generative function definition.
Diff code consists of Julia statements that can depend on non-diff code, but non-diff code cannot depend on the diff code.
To distinguish *diff code*  from regular code in the generative function, the `@diff` macro is placed in front of the statement, e.g.:
```julia
x = y + 1
@diff foo = 2
y = x
```
Diff code is only executed during update methods such as `update`, `fix_update`, or `extend` methods.
In other methods that are not associated with an update to a trace (e.g. `generate`, `simulate`, `assess`), the diff code is removed from the body of the generative function.
Therefore, the body of the generative function with the diff code removed must still be a valid generative function.

Unlike non-diff code, diff code has access to the argdiff value (using `@argdiff()`), and may invoke `@retdiff(<value>)`, which sets the retdiff value.
Diff code also has access to information about the change to the values of random choices and the change to the return values of calls to other generative functions.
Changes to the return values of random choices are queried using `@choicediff(<addr>)`, which must be invoked after the `@addr` expression for that address, and returns one of the following values:
```@docs
NewChoiceDiff
NoChoiceDiff
PrevChoiceDiff
```
Diff code also has access to the retdiff values associated with calls it makes to generative functions, using `@calldiff(<addr>)`, which returns a value of one of the following types:
```@docs
NewCallDiff
NoCallDiff
UnknownCallDiff
CustomCallDiff
```
Diff code can also pass argdiff values to generative functions that it calls, using the third argument in an `@addr` expression, which is always interpreted as diff code (depsite the absence of a `@diff` keyword).
```julia
@diff my_argdiff = @argdiff()
@diff argdiff_for_foo = ..
@addr(foo(arg1, arg2), addr, argdiff_for_foo)
@diff retdiff_from_foo = @calldiff(addr)
@diff @retdiff(..)
```

## Higher-Order Probabilistic Modules

## Using metaprogramming to implement new inference algorithms

Many Monte Carlo inference algorithms, like Hamiltonian Monte Carlo (HMC) and Metropolis-Adjusted Langevin Algorithm (MALA) are instances of general inference algorithm templates like Metropolis-Hastings, with specialized proposal distributions.
These algorithms can therefore be implemented with high-performance for a model if a compiled generative function defining the proposal is constructed manually.
However, it is also possible to write a generic implementation that automatically generates the generative function for the proposal using Julia's metaprogramming capabilities.
This section shows a simple example of writing a procedure that generates the code needed for a MALA update applied to an arbitrary set of top-level static addresses.

First, we write out a non-generic implementation of MALA.
MALA uses a proposal that tends to propose values in the direction of the gradient.
The procedure will be hardcoded to act on a specific set of addresses for a model called `model`.

```julia
set = DynamicAddressSet()
for addr in [:slope, :intercept, :inlier_std, :outlier_std]
    Gen.push_leaf_node!(set, addr)
end
mala_selection = StaticAddressSet(set)

@compiled @gen function mala_proposal(prev, tau)
    std::Float64 = sqrt(2*tau)
    gradients::StaticChoiceTrie = backprop_trace(model, prev, mala_selection, nothing)[3]
    @addr(normal(get_assignment(prev)[:slope] + tau * gradients[:slope], std), :slope)
    @addr(normal(get_assignment(prev)[:intercept] + tau * gradients[:intercept], std), :intercept)
    @addr(normal(get_assignment(prev)[:inlier_std] + tau * gradients[:inlier_std], std), :inlier_std)
    @addr(normal(get_assignment(prev)[:outlier_std] + tau * gradients[:outlier_std], std), :outlier_std)
end

mala_move(trace, tau::Float64) = mh(model, mala_proposal, (tau,), trace)
```

Next, we write a generic version that takes a set of addresses and generates the implementation for that set.
This version only works on a set of static top-level addresses.

```julia
function generate_mala_move(model, addrs)

    # create selection
    set = DynamicAddressSet()
    for addr in addrs
        Gen.push_leaf_node!(set, addr)
    end
    selection = StaticAddressSet(set)

    # generate proposal function
    stmts = Expr[]
    for addr in addrs
        quote_addr = QuoteNode(addr)
        push!(stmts, :(
            @addr(normal(get_assignment(prev)[$quote_addr] + tau * gradients[$quote_addr], std),
                  $quote_addr)
        ))
    end
    mala_proposal_name = gensym("mala_proposal")
    mala_proposal = eval(quote
        @compiled @gen function $mala_proposal_name(prev, tau)
            gradients::StaticChoiceTrie = backprop_trace(
                model, prev, $(QuoteNode(selection)), nothing)[3]
            std::Float64 = sqrt(2*tau)
            $(stmts...)
        end
    end)

    return (trace, tau::Float64) -> mh(model, mala_proposal, (tau,), trace)
end

```

We can then use the generate the MALA move for a speciic model and specific addresses using:
```julia
mala_move = generate_mala_move(model, [:slope, :intercept, :inlier_std, :outlier_std])
```
