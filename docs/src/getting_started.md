# Getting Started

## Installation

First, obtain Julia 1.0 or later, available [here](https://julialang.org/downloads/).

The Gen package can be installed with the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and then run:
```
pkg> add https://github.com/probcomp/Gen
```

## Quick Start

There are three main components to a typical Gen program.
First, we define a generative model, which are like Julia functions, but extended with some extra syntax.
The model below samples slope and intercept parameters, and then samples a y-coordinate for each of the x-coordinates that it takes as input.

```julia
using Gen

@gen function my_model(xs::Vector{Float64})
    slope = @addr(normal(0, 2), :slope)
    intercept = @addr(normal(0, 10), :intercept)
    for (i, x) in enumerate(xs)
        @addr(normal(slope * x + intercept, 1), "y-$i")
    end
end
```

Then, we write an inference program that implements an algorithm for manipulating the execution traces of the model.
Inference programs are regular Julia code.
The inference program below takes a data set, and runs a simple MCMC algorithm to fit slope and intercept parameters:

```julia
function my_inference_program(xs::Vector{Float64}, ys::Vector{Float64}, num_iters::Int)
    constraints = DynamicAssignment()
    for (i, y) in enumerate(ys)
        constraints["y-$i"] = y
    end
    (trace, _) = initialize(my_model, (xs,), constraints)
    slope_selection = select(:slope)
    intercept_selection = select(:intercept)
    for iter=1:num_iters
        (trace, _) = default_mh(trace, slope_selection)
        (trace, _) = default_mh(trace, intercept_selection)
    end
    assmt = get_assmt(trace)
    return (assmt[:slope], assmt[:intercept])
end
```

Finally, we run the inference program on some data, and get the results:

```julia
xs = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
ys = [8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90]
(slope, intercept) = my_inference_program(xs, ys, 1000)
println("slope: $slope, intercept: $slope")
```

## Visualization Framework

Because inference programs are regular Julia code, users can use whatever visualization or plotting libraries from the Julia ecosystem that they want.
However, we have paired Gen with the [GenViz](https://github.com/probcomp/GenViz) package, which is specialized for visualizing the output and operation of inference algorithms written in Gen.
