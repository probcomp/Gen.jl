using Gen
import Random

include("static_collapsed_model.jl")
include("dataset.jl")

@gen (static) function slope_proposal(trace)
    slope = trace[:slope]
    @trace(normal(slope, 0.5), :slope)
end

@gen (static) function intercept_proposal(trace)
    intercept = trace[:intercept]
    @trace(normal(intercept, 0.5), :intercept)
end

@gen (static) function inlier_std_proposal(trace)
    log_inlier_std = trace[:log_inlier_std]
    @trace(normal(log_inlier_std, 0.5), :log_inlier_std)
end

@gen (static) function outlier_std_proposal(trace)
    log_outlier_std = trace[:log_outlier_std]
    @trace(normal(log_outlier_std, 0.5), :log_outlier_std)
end

Gen.load_generated_functions()

function do_inference(xs, ys, num_iters)
    observations = choicemap()
    for (i, y) in enumerate(ys)
        observations[:data => i => :z] = y
    end

    # initial trace
    (trace, _) = generate(model, (xs,), observations)

    scores = Vector{Float64}(undef, num_iters)
    for i=1:num_iters

        # steps on the parameters
        (trace, _) = metropolis_hastings(trace, slope_proposal, ())
        (trace, _) = metropolis_hastings(trace, intercept_proposal, ())
        (trace, _) = metropolis_hastings(trace, inlier_std_proposal, ())
        (trace, _) = metropolis_hastings(trace, outlier_std_proposal, ())

        score = get_score(trace)
        scores[i] = score

        # print
        slope = trace[:slope]
        intercept = trace[:intercept]
        inlier_std = exp(trace[:log_inlier_std])
        outlier_std = exp(trace[:log_outlier_std])
        println("score: $score, slope: $slope, intercept: $intercept, inlier_std: $inlier_std, outlier_std: $outlier_std")
    end
    return scores
end

(xs, ys) = make_data_set(200)
do_inference(xs, ys, 10)
@time scores = do_inference(xs, ys, 100)
