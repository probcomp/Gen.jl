# [Getting Started: Linear Regression](@id getting_started)

Let's write a short Gen program that does Bayesian linear regression - that is, given a set of observation points in the xy-plane, we want to find a line that fits them "well". What does "well" mean in Bayesian linear regression? Well one way of interpreting this is by proposing a line that makes the data highly likely - as in a high probability of occuring. 

First, we need to define a _generative model_ that describes how we believe the points were generated.

```math
\mu \sim N(0,2)\\
b \sim N(0,10)\\
\epsilon_i \sim N(0,1)\\
y_i | x_i \sim \mu x_i + b + \epsilon_i
```

This model first randomly samples a slope ``\mu`` and an intercept ``b`` from normal distributions to define the line ``y=mx+b``. Next each x-coordinate is evaluated and perturbed with a little noise. Now let's write this as a probabilistic program.

The description of the line is a mathematical one, but we can write it using normal code constructs. The _generative model_ is a Julia function with a _tilde_ (~) operator for sampling. Observe that the function below looks almost the same as the generative model.

```@example linear_regression
using Gen

@gen function my_model(xs::Vector{Float64})
    slope ~ normal(0, 2)
    intercept ~ normal(0, 10)
    for (i, x) in enumerate(xs)
        {"y-$i"} ~ normal(slope * x + intercept, 1)
    end
end
nothing # hide
```

Second, we write an _inference program_ that implements an algorithm for manipulating the execution traces of the model.
Inference programs are regular Julia code, and make use of Gen's standard inference library.

The inference program below takes in a data set, and runs an iterative MCMC algorithm to fit `slope` and `intercept` parameters:

```@example linear_regression
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
    for _=1:num_iters
        (trace, _) = metropolis_hastings(trace, select(:slope))
        (trace, _) = metropolis_hastings(trace, select(:intercept))
    end
    
    # From the final trace, read out the slope and
    # the intercept.
    choices = get_choices(trace)
    return (choices[:slope], choices[:intercept])
end
nothing # hide
```

Finally, we run the inference program on some data, and get the results:

```@example linear_regression
xs = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
ys = [8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90]
(slope, intercept) = my_inference_program(xs, ys, 1000)
println("slope: $slope, intercept: $intercept")
```
