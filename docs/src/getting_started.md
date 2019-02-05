# Getting Started

## Installation

First, obtain Julia 1.0 or later, available [here](https://julialang.org/downloads/).

The Gen package can be installed with the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and then run:
```
pkg> add https://github.com/probcomp/Gen
```
To test the installation, run the quick start example in the next section, or run the tests with:
```julia
using Pkg; Pkg.test("Gen")
```

## Quick Start

Let's write a short Gen program that does Bayesian linear regression: given a set of points in the (x, y) plane, we want to find a line that fits them well.

There are three main components to a typical Gen program.

First, we define a _generative model_: a Julia function, extended with some extra syntax, that, conceptually, simulates a fake dataset. The model below samples `slope` and `intercept` parameters, and then for each of the x-coordinates that it accepts as input, samples a corresponding y-coordinate. We name the random choices we make with `@addr`, so we can refer to them in our inference program.

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

Second, we write an _inference program_ that implements an algorithm for manipulating the execution traces of the model.
Inference programs are regular Julia code, and make use of Gen's standard inference library.

The inference program below takes in a data set, and runs an iterative MCMC algorithm to fit `slope` and `intercept` parameters:

```julia
function my_inference_program(xs::Vector{Float64}, ys::Vector{Float64}, num_iters::Int)
    # Create a set of constraints fixing the 
    # y coordinates to the observed y values
    constraints = choicemap()
    for (i, y) in enumerate(ys)
        constraints["y-$i"] = y
    end
    
    # Run the model, constrained by `constraints`,
    # to get an initial execution trace
    (trace, _) = generate(my_model, (xs,), constraints)
    
    # Iteratively update the slope then the intercept,
    # using Gen's metropolis_hastings operator.
    for iter=1:num_iters
        (trace, _) = metropolis_hastings(trace, select(:slope))
        (trace, _) = metropolis_hastings(trace, select(:intercept))
    end
    
    # From the final trace, read out the slope and
    # the intercept.
    choices = get_choices(trace)
    return (choices[:slope], choices[:intercept])
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

An example demonstrating the use of GenViz for this Quick Start linear regression problem is available in the [gen-examples](https://github.com/probcomp/gen-examples) repository. The code there is mostly the same as above, with a few small changes to incorporate an animated visualization of the inference process:

1. It starts a visualization server and initializes a visualization before performing inference:
```julia
# Start a visualization server on port 8000
server = VizServer(8000)

# Initialize a visualization with some parameters
viz = Viz(server, joinpath(@__DIR__, "vue/dist"), Dict("xs" => xs, "ys" => ys, "num" => length(xs), "xlim" => [minimum(xs), maximum(xs)], "ylim" => [minimum(ys), maximum(ys)]))

# Open the visualization in a browser
openInBrowser(viz)
```

The `"vue/dist"` is a path to a custom _trace renderer_ that draws the (x, y) points and the line represented by a trace; see the GenViz documentation for more details. The code for the renderer is [here](https://github.com/probcomp/gen-examples/blob/master/quickstart/vue/src/components/Trace.vue).

2. It passes the visualization object into the inference program.
```julia
(slope, intercept) = my_inference_program(xs, ys, 1000000, viz)
```

3. In the inference program, it puts the current trace into the visualization at each iteration:
```julia
for iter=1:num_iters
    putTrace!(viz, 1, trace_to_dict(trace))
    (trace, _) = metropolis_hastings(trace, select(:slope))
    (trace, _) = metropolis_hastings(trace, select(:intercept))
end
```
