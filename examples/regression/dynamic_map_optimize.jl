using Gen
import Random

include("dynamic_model.jl")
include("dataset.jl")

@gen function is_outlier_proposal(prev, i::Int)
    prev = get_assignment(prev)[:data => i => :z]
    @addr(bernoulli(prev ? 0.0 : 1.0), :data => i => :z)
end

slope_intercept_selection = DynamicAddressSet()
push!(slope_intercept_selection, :slope)
push!(slope_intercept_selection, :intercept)

std_selection = DynamicAddressSet()
push!(std_selection, :log_inlier_std)
push!(std_selection, :log_outlier_std)

function do_inference(xs, ys, num_iters)
    observations = DynamicAssignment()
    for (i, y) in enumerate(ys)
        observations[:data => i => :y] = y
    end
    observations[:log_inlier_std] = 0.
    observations[:log_outlier_std] = 0.
 
    # initial trace
    (trace, _) = generate(model, (xs,), observations)

    scores = Vector{Float64}(undef, num_iters)
    for i=1:num_iters
        trace = map_optimize(model, slope_intercept_selection, trace, max_step_size=1., min_step_size=1e-10)
        trace = map_optimize(model, std_selection, trace, max_step_size=1., min_step_size=1e-10)
    
        # step on the outliers
        for j=1:length(xs)
            trace = mh(model, is_outlier_proposal, (j,), trace)
        end
    
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
iters = 100
@time do_inference(xs, ys, iters)
@time scores = do_inference(xs, ys, iters)
println(scores)

using PyPlot

figure(figsize=(4, 2))
plot(scores)
ylabel("Log probability density")
xlabel("Iterations")
tight_layout()
savefig("dynamic_map_optimize_scores.png")
