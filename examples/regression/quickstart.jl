using Gen

@gen function my_model(xs)
    slope = @trace(normal(0, 2), :slope)
    intercept = @trace(normal(0, 10), :intercept)
    for (i, x) in enumerate(xs)
        @trace(normal(slope * x + intercept, 1), "y-$i")
    end
end

function my_inference_program(xs, ys, num_iters)
    constraints = choicemap()
    for (i, y) in enumerate(ys)
        constraints["y-$i"] = y
    end
    (trace, _) = generate(my_model, (xs,), constraints)
    slope_selection = select(:slope)
    intercept_selection = select(:intercept)
    for iter=1:num_iters
        (trace, _) = metropolis_hastings(trace, slope_selection)
        (trace, _) = metropolis_hastings(trace, intercept_selection)
    end
    return (trace[:slope], trace[:intercept])
end

xs = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
ys = [8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90]
(slope, intercept) = my_inference_program(xs, ys, 1000)
println("slope: $slope, intercept: $intercept")
