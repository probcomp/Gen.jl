using Gen
import Random
using FunctionalCollections: PersistentVector

include("static_model.jl")
include("dataset.jl")

@gen (static) function is_outlier_proposal(prev, i::Int)
    prev_z::Bool = get_assmt(prev)[:data => i => :z]
    @addr(bernoulli(prev_z ? 0.0 : 1.0), :data => i => :z)
end

Gen.load_generated_functions()

line_selection = StaticAddressSet(select(:slope, :intercept))
std_selection = StaticAddressSet(select(:log_inlier_std, :log_outlier_std))

function do_inference(xs, ys, num_iters)

    observations = DynamicAssignment()
    for (i, y) in enumerate(ys)
        observations[:data => i => :y] = y
    end
    observations[:log_inlier_std] = 0.
    observations[:log_outlier_std] = 0.
    
    # initial trace
    (trace, _) = initialize(model, (xs,), observations)
    
    scores = Vector{Float64}(undef, num_iters)
    for i=1:num_iters
        (trace, _) = mala(trace, line_selection, 0.0001)
        (trace, _) = mala(trace, std_selection, 0.0001)
    
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
@time scores = do_inference(xs, ys, 1000)
