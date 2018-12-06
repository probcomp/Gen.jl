using Gen
import Random
using FunctionalCollections: PersistentVector

include("static_model.jl")
include("dataset.jl")

@staticgen function flip_z(z::Bool)
    @addr(bernoulli(z ? 0.0 : 1.0), :z)
end

data_proposal = at_dynamic(flip_z, Int)

@staticgen function is_outlier_proposal(prev, i::Int)
    prev_z::Bool = get_assignment(prev)[:data => i => :z]
    @addr(data_proposal(i, (prev_z,)), :data) 
end

Gen.load_generated_functions()

line_selection = let
    s = DynamicAddressSet()
    push!(s, :slope)
    push!(s, :intercept)
    StaticAddressSet(s)
end

std_selection = let
    s = DynamicAddressSet()
    push!(s, :log_inlier_std)
    push!(s, :log_outlier_std)
    StaticAddressSet(s)
end

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
        trace = mala(model, line_selection, trace, 0.0001)
        trace = mala(model, std_selection, trace, 0.0001)
    
        # step on the outliers
        for j=1:length(xs)
            trace = custom_mh(model, is_outlier_proposal, (j,), trace)
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
@time scores = do_inference(xs, ys, 1000)
println(scores)

using PyPlot
figure(figsize=(4, 2))
plot(scores)
ylabel("Log probability density")
xlabel("Iterations")
tight_layout()
savefig("static_mala_scores.png")
