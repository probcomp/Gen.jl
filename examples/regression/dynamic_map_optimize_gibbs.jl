using Gen
import Random

include("dynamic_model.jl")
include("dataset.jl")

@gen function gibbs_proposal(trace, i::Int)
    args = get_args(trace)
    constraints = choicemap()
    constraints[:data => i => :z] = false
    (_, weight1) = update(trace, args, map((_) -> NoChange(), args), constraints)
    constraints[:data => i => :z] = true
    (_, weight2) = update(trace, args, map((_) -> NoChange(), args), constraints)
    prob_true = exp(weight2 - logsumexp([weight1, weight2]))
    @trace(bernoulli(prob_true), :data => i => :z)
end

slope_intercept_selection = select(:slope, :intercept)
std_selection = select(:log_inlier_std, :log_outlier_std)

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

        # gibbs step on the outliers
        for j=1:length(xs)
            (trace, _) = metropolis_hastings(trace, gibbs_proposal, (j,))
        end

        score = get_score(trace)
        scores[i] = score

        # print
        assignment = get_choices(trace)
        slope = assignment[:slope]
        intercept = assignment[:intercept]
        inlier_std = exp(assignment[:log_inlier_std])
        outlier_std = exp(assignment[:log_outlier_std])
        println("score: $score, slope: $slope, intercept: $intercept, inlier_std: $inlier_std, outlier_std: $outlier_std")
    end
    return scores
end

(xs, ys) = make_data_set(200)
do_inference(xs, ys, 10)
@time scores = do_inference(xs, ys, 50)
