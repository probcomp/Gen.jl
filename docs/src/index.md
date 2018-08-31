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
├── "intercept" : 0.3706308263323224
│
├── "slope" : 2.5143135249129553
│
├── 2
│   │
│   └── "y" : -9.326849685997791
│
├── 3
│   │
│   └── "y" : 0.6270355023754557
│
├── 4
│   │
│   └── "y" : 8.45301143629185
│
└── 1
    │
    └── "y" : -13.10634076245438
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
