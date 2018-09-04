# Gen Introduction

Gen is an extensible and reasonably performant probabilistic computing platform that makes it easier to develop probabilistic inference and learning applications.

## Generative Functions, Traces, and Choice Tries

Stochastic generative processes are represented in Gen as *generative functions*.
Generative functions are Julia functions that have been annotated using the `@gen` macro.
The generative function below takes a vector of x-coordinates, randomly generates the slope and intercept parameters of a line, and returns a randomly vector of y-coordinates sampled near that line, at the given x-coordinates:

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
However, what distinguishes generative functions from plain-old simulators is their 
 ability to be *traced*.
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
trace = simulate(model, (xs,))
```

The trace that is returned is a form of stack trace of the generative function that contains, among other things, the values for the random choices that were annotated with `@addr`.
To extract the values of the random choices from a trace, we use the method `get_choices`:

```julia
choices = get_choices(trace)
```

The `get_choices` method returns a *choice trie*, which is a trie (prefix tree) that contains the values of random choices.
Printing the choice trie gives a pretty printed representation:

```julia
print(choices)
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

Generative functions can also call other generative functions, these calls can also be traced using `@addr`:

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

This results in a hierarchical choice trie:

```julia
trace = simulate(model, (xs,))
choices = get_choices(trace)
print(choices)
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

We can read values from a choice trie using the following syntax:
```julia
choices["intercept"]
```

To read the value at a hierarchical address, we provide a `Pair` where the first element of the pair is the first part ofthe hierarchical address, and the second element is the rest of the address.
For example:
```julia
choices[1 => "y"]
```

Julia provides the operator `=>` for constructing `Pair` values.
Long hierarchical addresses can be constructed by chaining this operator, which associates right:
```julia
choices[1 => "y" => :foo => :bar]
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
trace = simulate(model, (xs,))
choices = get_choices(trace)
print(choices)
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

## Change Propagation

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
    @addr(normal(get_choices(prev)[:slope] + tau * gradients[:slope], std), :slope)
    @addr(normal(get_choices(prev)[:intercept] + tau * gradients[:intercept], std), :intercept)
    @addr(normal(get_choices(prev)[:inlier_std] + tau * gradients[:inlier_std], std), :inlier_std)
    @addr(normal(get_choices(prev)[:outlier_std] + tau * gradients[:outlier_std], std), :outlier_std)
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
            @addr(normal(get_choices(prev)[$quote_addr] + tau * gradients[$quote_addr], std),
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
