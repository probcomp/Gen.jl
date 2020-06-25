module DynamicMHBenchmark
using Gen
import Random

include("../../examples/regression/dynamic_model.jl")
include("../../examples/regression/dataset.jl")

@gen function slope_proposal(trace)
    slope = trace[:slope]
    @trace(normal(slope, 0.5), :slope)
end

@gen function intercept_proposal(trace)
    intercept = trace[:intercept]
    @trace(normal(intercept, 0.5), :intercept)
end

@gen function inlier_std_proposal(trace)
    log_inlier_std = trace[:log_inlier_std]
    @trace(normal(log_inlier_std, 0.5), :log_inlier_std)
end

@gen function outlier_std_proposal(trace)
    log_outlier_std = trace[:log_outlier_std]
    @trace(normal(log_outlier_std, 0.5), :log_outlier_std)
end

@gen function is_outlier_proposal(trace, i::Int)
    prev = trace[:data => i => :z]
    @trace(bernoulli(prev ? 0.0 : 1.0), :data => i => :z)
end

function do_inference(xs, ys, num_iters)
    observations = choicemap()
    for (i, y) in enumerate(ys)
        observations[:data => i => :y] = y
    end

    # initial trace
    (trace, _) = generate(model, (xs,), observations)

    scores = Vector{Float64}(undef, num_iters)
    for i=1:num_iters

        # steps on the parameters
        for j=1:5
            (trace, _) = metropolis_hastings(trace, slope_proposal, ())
            (trace, _) = metropolis_hastings(trace, intercept_proposal, ())
            (trace, _) = metropolis_hastings(trace, inlier_std_proposal, ())
            (trace, _) = metropolis_hastings(trace, outlier_std_proposal, ())
        end

        # step on the outliers
        for j=1:length(xs)
            (trace, _) = metropolis_hastings(trace, is_outlier_proposal, (j,))
        end

        score = get_score(trace)
        scores[i] = score

        # print
        slope = trace[:slope]
        intercept = trace[:intercept]
        inlier_std = exp(trace[:log_inlier_std])
        outlier_std = exp(trace[:log_outlier_std])
    end
    return scores
end

println("Simple dynamic DSL MH on regression model:")
(xs, ys) = make_data_set(200)
do_inference(xs, ys, 10)
@time do_inference(xs, ys, 20)
@time do_inference(xs, ys, 20)
println()

end