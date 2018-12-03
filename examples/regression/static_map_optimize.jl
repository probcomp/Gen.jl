using Gen
import Random
using FunctionalCollections: PersistentVector

using Flux
import Flux
using Flux.Tracker: @grad, TrackedReal, TrackedArray, track
import Flux.Tracker: param

#########
# model #
#########

param(value::Bool) = value

param(value::Int) = value

Base.fill(a::TrackedReal, b::Integer) = track(fill, a, b)

@grad function Base.fill(a, b)
    fill(Flux.Tracker.data(a), Flux.Tracker.data(b)), grad -> (sum(grad), nothing)
end

@staticgen function datum(x::Float64, @grad(inlier_std::Float64), @grad(outlier_std::Float64),
                          @grad(slope::Float64), @grad(intercept::Float64))
    is_outlier::Bool = @addr(bernoulli(0.5), :z)
    std = is_outlier ? inlier_std : outlier_std
    y::Float64 = @addr(normal(x * slope + intercept, std), :y)
    return y
end

data = Map(datum)

function compute_argdiff(inlier_std_diff, outlier_std_diff, slope_diff, intercept_diff)
    if all([c == NoChoiceDiff() for c in [
            inlier_std_diff, outlier_std_diff, slope_diff, intercept_diff]])
        noargdiff
    else
        unknownargdiff
    end
end

@staticgen function model(xs::Vector{Float64})
    n = length(xs)
    inlier_log_std::Float64 = @addr(normal(0, 2), :log_inlier_std)
    outlier_log_std::Float64 = @addr(normal(0, 2), :log_outlier_std)
    inlier_std = exp(inlier_log_std)
    outlier_std = exp(outlier_log_std)
    slope::Float64 = @addr(normal(0, 2), :slope)
    intercept::Float64 = @addr(normal(0, 2), :intercept)
    @diff inlier_std_diff = @choicediff(:log_inlier_std)
    @diff outlier_std_diff = @choicediff(:log_outlier_std)
    @diff slope_diff = @choicediff(:slope)
    @diff intercept_diff = @choicediff(:intercept)
    @diff argdiff = compute_argdiff(
            inlier_std_diff, outlier_std_diff, slope_diff, intercept_diff)
    @addr(data(xs, fill(inlier_std, n), fill(outlier_std, n),
               fill(slope, n), fill(intercept, n)),
          :data, argdiff)
end

#######################
# inference operators #
#######################

@staticgen function flip_z(z::Bool)
    @addr(bernoulli(z ? 0.0 : 1.0), :z)
end

data_proposal = at_dynamic(flip_z, Int)

@staticgen function is_outlier_proposal(prev, i::Int)
    prev_z::Bool = get_assignment(prev)[:data => i => :z]
    @addr(data_proposal(i, (prev_z,)), :data) 
end

Gen.load_generated_functions()

#####################
# generate data set #
#####################

Random.seed!(1)

prob_outlier = 0.5
true_inlier_noise = 0.5
true_outlier_noise = 5.0
true_slope = -1
true_intercept = 2
xs = collect(range(-5, stop=5, length=200))
ys = Float64[]
for (i, x) in enumerate(xs)
    if rand() < prob_outlier
        y = true_slope * x + true_intercept + randn() * true_inlier_noise
    else
        y = true_slope * x + true_intercept + randn() * true_outlier_noise
    end
    push!(ys, y)
end

##################
# run experiment #
##################

slope_intercept_selection = let
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

function do_inference(n)

    observations = DynamicAssignment()
    for (i, y) in enumerate(ys)
        observations[:data => i => :y] = y
    end
    observations[:log_inlier_std] = 0.
    observations[:log_outlier_std] = 0.

    # initial trace
    (trace, _) = generate(model, (xs,), observations)

    scores = Vector{Float64}(undef, n)
    for i=1:n
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

iters = 100
@time do_inference(iters)
@time scores = do_inference(iters)

using PyPlot

figure(figsize=(4, 2))
plot(scores)
ylabel("Log probability density")
xlabel("Iterations")
tight_layout()
savefig("scores.png")
