using Gen
import Random

include("dynamic_model.jl")
include("dataset.jl")

@gen function is_outlier_proposal(prev, i::Int)
    prev = get_assmt(prev)[:data => i => :z]
    @addr(bernoulli(prev ? 0.0 : 1.0), :data => i => :z)
end

function do_inference(xs, ys, num_iters)
    observations = DynamicAssignment()
    for (i, y) in enumerate(ys)
        observations[:data => i => :y] = y
    end

    slope_sel = DynamicAddressSet()
    push!(slope_sel, :slope)

    intercept_sel = DynamicAddressSet()
    push!(intercept_sel, :intercept)

    inlier_std_sel = DynamicAddressSet()
    push!(inlier_std_sel, :log_inlier_std)

    outlier_std_sel = DynamicAddressSet()
    push!(outlier_std_sel, :log_outlier_std)
    
    # initial trace
    (trace, _) = initialize(model, (xs,), observations)
    
    scores = Vector{Float64}(undef, num_iters)
    for i=1:num_iters
    
        # steps on the parameters
        for j=1:5
            (trace, _) = default_mh(trace, slope_sel)
            (trace, _) = default_mh(trace, intercept_sel)
            (trace, _) = default_mh(trace, inlier_std_sel)
            (trace, _) = default_mh(trace, outlier_std_sel)
        end
    
        # step on the outliers
        for j=1:length(xs)
            (trace, _) = custom_mh(trace, is_outlier_proposal, (j,))
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
