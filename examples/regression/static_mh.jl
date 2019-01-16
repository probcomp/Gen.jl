using Gen
import Random

include("static_model.jl")
include("dataset.jl")

@gen (static) function slope_proposal(prev)
    slope = get_assmt(prev)[:slope]
    @addr(normal(slope, 0.5), :slope)
end

@gen (static) function intercept_proposal(prev)
    intercept = get_assmt(prev)[:intercept]
    @addr(normal(intercept, 0.5), :intercept)
end

@gen (static) function inlier_std_proposal(prev)
    log_inlier_std = get_assmt(prev)[:log_inlier_std]
    @addr(normal(log_inlier_std, 0.5), :log_inlier_std)
end

@gen (static) function outlier_std_proposal(prev)
    log_outlier_std = get_assmt(prev)[:log_outlier_std]
    @addr(normal(log_outlier_std, 0.5), :log_outlier_std)
end

@gen (static) function flip_z(z::Bool)
    @addr(bernoulli(z ? 0.0 : 1.0), :z)
end

@gen (static) function is_outlier_proposal(prev, i::Int)
    prev_z::Bool = get_assmt(prev)[:data => i => :z]
    @addr(bernoulli(prev_z ? 0.0 : 1.0), :data => i => :z)
end

Gen.load_generated_functions()

function do_inference(xs, ys, num_iters)
    observations = DynamicAssignment()
    for (i, y) in enumerate(ys)
        observations[:data => i => :y] = y
    end

    # initial trace
    (trace, _) = initialize(model, (xs,), observations)

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
        assignment = get_assmt(trace)
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
