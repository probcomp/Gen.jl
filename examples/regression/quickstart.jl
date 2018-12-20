using Gen

@gen function my_model(xs)
    slope = @addr(normal(0, 2), :slope)
    intercept = @addr(normal(0, 10), :intercept)
    for (i, x) in enumerate(xs)
        @addr(normal(slope * x + intercept, 1), "y-$i")
    end
end

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

xs = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
ys = [8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90]
(slope, intercept) = my_inference_program(xs, ys, 1000)
println("slope: $slope, intercept: $intercept")
