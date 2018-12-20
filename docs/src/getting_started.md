# Getting Started

## Installation

First, obtain Julia 1.0 or later, available [here](https://julialang.org/downloads/).

The Gen package can be installed with the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and then run:
```
pkg> add https://github.com/probcomp/Gen
```

(Optional) Install the GenViz package:
```
pkg> add https://github.com/probcomp/GenViz
```

## Example

```jldoctest
using Gen

# 1. define a generative model

@gen function my_model(xs)
    slope = @addr(normal(0, 2), :slope)
    intercept = @addr(normal(0, 10), :intercept)
    ys = Vector{Float64}(undef, length(xs))
    for (i, x) in enumerate(xs)
        ys[i] = @addr(normal(slope * x + intercept, 1), "y-$i")
    end
end

# 2. define an inference program

function my_inference_program(xs, ys, num_iters)
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

# 3. run inference program for a particular data set

xs = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
ys = [8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90]
(slope, intercept) = my_inference_program(xs, ys, 1000)
# output
```
