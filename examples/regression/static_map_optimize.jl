using Gen
import Random

include("static_model.jl")
include("dataset.jl")

@gen (static) function is_outlier_proposal(trace, i::Int)
    prev_z::Bool = trace[:data => i => :z]
    @trace(bernoulli(prev_z ? 0.0 : 1.0), :data => i => :z)
end

Gen.load_generated_functions()

slope_intercept_selection = StaticSelection(select(:slope, :intercept))
std_selection = StaticSelection(select(:log_inlier_std, :log_outlier_std))

function do_inference(xs, ys, num_iters)

    observations = choicemap()
    for (i, y) in enumerate(ys)
        observations[:data => i => :y] = y
    end
    observations[:log_inlier_std] = 0.
    observations[:log_outlier_std] = 0.

    # initial trace
    (trace, _) = generate(model, (xs,), observations)

    scores = Vector{Float64}(undef, num_iters)
    for i=1:num_iters
        trace = map_optimize(trace, slope_intercept_selection, max_step_size=1., min_step_size=1e-10)
        trace = map_optimize(trace, std_selection, max_step_size=1., min_step_size=1e-10)

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
