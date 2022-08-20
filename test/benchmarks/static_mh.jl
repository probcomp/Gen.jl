module StaticMHBenchmark
using Gen
import Random
using Gen: ifelse

include("dataset.jl")

@gen (static) function slope_proposal(trace)
    slope = trace[:slope]
    @trace(normal(slope, 0.5), :slope)
end

@gen (static) function intercept_proposal(trace)
    intercept = trace[:intercept]
    @trace(normal(intercept, 0.5), :intercept)
end

@gen (static) function inlier_std_proposal(trace)
    log_inlier_std = trace[:log_inlier_std]
    @trace(normal(log_inlier_std, 0.5), :log_inlier_std)
end

@gen (static) function outlier_std_proposal(trace)
    log_outlier_std = trace[:log_outlier_std]
    @trace(normal(log_outlier_std, 0.5), :log_outlier_std)
end

@gen (static) function flip_z(z::Bool)
    @trace(bernoulli(z ? 0.0 : 1.0), :z)
end

@gen (static) function is_outlier_proposal(trace, i::Int)
    prev_z = trace[:data => i => :z]
    @trace(bernoulli(prev_z ? 0.0 : 1.0), :data => i => :z)
end

@gen (static) function is_outlier_proposal(trace, i::Int)
    prev_z = trace[:data => i => :z]
    @trace(bernoulli(prev_z ? 0.0 : 1.0), :data => i => :z)
end

@gen (static) function datum(x::Float64, (grad)(inlier_std::Float64), (grad)(outlier_std::Float64),
                             (grad)(slope::Float64), (grad)(intercept::Float64))
    is_outlier = @trace(bernoulli(0.5), :z)
    std = ifelse(is_outlier, inlier_std, outlier_std)
    y = @trace(normal(x * slope + intercept, std), :y)
    return y
end

data = Map(datum)

@gen (static) function model(xs::Vector{Float64})
    n = length(xs)
    inlier_log_std = @trace(normal(0, 2), :log_inlier_std)
    outlier_log_std = @trace(normal(0, 2), :log_outlier_std)
    inlier_std = exp(inlier_log_std)
    outlier_std = exp(outlier_log_std)
    slope = @trace(normal(0, 2), :slope)
    intercept = @trace(normal(0, 2), :intercept)
    @trace(data(xs, fill(inlier_std, n), fill(outlier_std, n),
        fill(slope, n), fill(intercept, n)), :data)
end

function do_inference(xs, ys, num_iters)
    observations = choicemap()
    for (i, y) in enumerate(ys)
        observations[:data => i => :y] = y
    end

    # initial trace
    (trace, _) = generate(model, (xs,), observations)

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
        slope = trace[:slope]
        intercept = trace[:intercept]
        inlier_std = exp(trace[:log_inlier_std])
        outlier_std = exp(trace[:log_outlier_std])
    end
    return scores
end

(xs, ys) = make_data_set(200)
do_inference(xs, ys, 10)
println("Simple static DSL (including CallAt nodes) MH on regression model:")
@time do_inference(xs, ys, 50)
@time do_inference(xs, ys, 50)
println()
end
