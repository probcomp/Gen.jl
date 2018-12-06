using Gen
import Random

include("static_model.jl")
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

@staticgen function flip_z(z::Bool)
    @addr(bernoulli(z ? 0.0 : 1.0), :z)
end

data_proposal = at_dynamic(flip_z, Int)

@staticgen function is_outlier_proposal(prev, i::Int)
    prev_z::Bool = get_assignment(prev)[:data => i => :z]
    @addr(data_proposal(i, (prev_z,)), :data) 
end

Gen.load_generated_functions()

function do_inference(xs, ys, num_iters)
    observations = DynamicAssignment()
    for (i, y) in enumerate(ys)
        observations[:data => i => :y] = y
    end

    # initial trace
    (trace, _) = generate(model, (xs,), observations)

    scores = Vector{Float64}(undef, num_iters)
    for i=1:num_iters

        # steps on the parameters
        for j=1:5
            trace = mh(model, slope_proposal, (), trace)
            trace = mh(model, intercept_proposal, (), trace)
            trace = mh(model, inlier_std_proposal, (), trace)
            trace = mh(model, outlier_std_proposal, (), trace)
        end

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
do_inference(xs, ys, 10)
@time scores = do_inference(xs, ys, 50)
println(scores)

using PyPlot

figure(figsize=(4, 2))
plot(scores)
ylabel("Log probability density")
xlabel("Iterations")
tight_layout()
savefig("static_mh_scores.png")
