using Gen
import Random

include("dynamic_model.jl")
include("dataset.jl")

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
            (trace, _) = metropolis_hastings(trace, select(:slope))
            (trace, _) = metropolis_hastings(trace, select(:intercept))
            (trace, _) = metropolis_hastings(trace, select(:log_inlier_std))
            (trace, _) = metropolis_hastings(trace, select(:log_outlier_std))
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
        println("score: $score, slope: $slope, intercept: $intercept, inlier_std: $inlier_std, outlier_std: $outlier_std")
    end
    return scores
end

(xs, ys) = make_data_set(200)
do_inference(xs, ys, 10)
@time scores = do_inference(xs, ys, 50)
