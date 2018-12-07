using Gen
import Random

include("static_collapsed_model.jl")
include("dataset.jl")

@staticgen function slope_proposal(prev)
    slope = get_assignment(prev)[:slope]
    @addr(normal(slope, 0.5), :slope)
end

@staticgen function intercept_proposal(prev)
    intercept = get_assignment(prev)[:intercept]
    @addr(normal(intercept, 0.5), :intercept)
end

@staticgen function inlier_std_proposal(prev)
    log_inlier_std = get_assignment(prev)[:log_inlier_std]
    @addr(normal(log_inlier_std, 0.5), :log_inlier_std)
end

@staticgen function outlier_std_proposal(prev)
    log_outlier_std = get_assignment(prev)[:log_outlier_std]
    @addr(normal(log_outlier_std, 0.5), :log_outlier_std)
end

Gen.load_generated_functions()

function do_inference(xs, ys, num_iters)
    observations = DynamicAssignment()
    for (i, y) in enumerate(ys)
        observations[:data => i] = y
    end

    # initial trace
    (trace, _) = initialize(model, (xs,), observations)

    scores = Vector{Float64}(undef, num_iters)
    for i=1:num_iters

        # steps on the parameters
        trace = custom_mh(model, slope_proposal, (), trace)
        trace = custom_mh(model, intercept_proposal, (), trace)
        trace = custom_mh(model, inlier_std_proposal, (), trace)
        trace = custom_mh(model, outlier_std_proposal, (), trace)

        score = get_call_record(trace).score
        scores[i] = score

        # print
        assignment = get_assignment(trace)
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
@time scores = do_inference(xs, ys, 100)
println(scores)

using PyPlot

figure(figsize=(4, 2))
plot(scores)
ylabel("Log probability density")
xlabel("Iterations")
tight_layout()
savefig("static_mh_collapsed_scores.png")
