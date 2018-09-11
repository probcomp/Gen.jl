using Gen
import Random
using FunctionalCollections

#####################
# uncollapsed model #
#####################

struct Params
    prob_outlier::Float64
    inlier_std::Float64
    outlier_std::Float64
    slope::Float64
    intercept::Float64
end

@compiled @gen function datum(x::Float64, params::Params)
    is_outlier::Bool = @addr(bernoulli(params.prob_outlier), :z)
    std::Float64 = is_outlier ? params.inlier_std : params.outlier_std
    y::Float64 = @addr(normal(x * params.slope + params.intercept, std), :y)
    return y
end

data = plate(datum)

function compute_data_change(inlier_std_change, outlier_std_change, slope_change, intercept_change)
    if all([c !== nothing && (c == NoChange() || !c[1]) for c in [
            inlier_std_change, outlier_std_change, slope_change, intercept_change]])
        NoChange()
    else
        nothing
    end
end

@compiled @gen function model(xs::Vector{Float64})
    n::Int = length(xs)
    inlier_std::Float64 = @addr(gamma(1, 1), :inlier_std)
    outlier_std::Float64 = @addr(gamma(1, 1), :outlier_std)
    slope::Float64 = @addr(normal(0, 2), :slope)
    intercept::Float64 = @addr(normal(0, 2), :intercept)
    params::Params = Params(0.5, inlier_std, outlier_std, slope, intercept)
    inlier_std_change::Union{Tuple{Bool,Float64},Nothing} = @change(:inlier_std)
    outlier_std_change::Union{Tuple{Bool,Float64},Nothing} = @change(:outlier_std)
    slope_change::Union{Tuple{Bool,Float64},Nothing} = @change(:slope)
    intercept_change::Union{Tuple{Bool,Float64},Nothing} = @change(:intercept)
    change::Union{NoChange,Nothing} = compute_data_change(
        inlier_std_change, outlier_std_change, slope_change, intercept_change)
    ys::PersistentVector{Float64} = @addr(data(xs, fill(params, n)), :data, change)
    return ys
end


###################
# collapsed model #
###################

import Distributions
import Gen: random, logpdf, get_static_argument_types
struct TwoNormals <: Distribution{Float64} end
const two_normals = TwoNormals()
function logpdf(::TwoNormals, x, mu, sigma1, sigma2)
    if sigma1 < 0 || sigma2 < 0
        return -Inf
    end
    l1 = Distributions.logpdf(Distributions.Normal(mu, sigma1), x) + log(0.5)
    l2 = Distributions.logpdf(Distributions.Normal(mu, sigma2), x) + log(0.5)
    m = max(l1, l2)
    m + log(exp(l1 - m) + exp(l2 - m))
end
function random(::TwoNormals, mu, sigma1, sigma2)
    mu + (rand() < 0.5 ? sigma1 : sigma2) * randn()
end
get_static_argument_types(::TwoNormals) = [Float64, Float64, Float64]

data_collapsed = plate(two_normals)

@compiled @gen function model_collapsed(xs::Vector{Float64})
    n::Int = length(xs)
    inlier_std::Float64 = @addr(gamma(1, 1), :inlier_std)
    outlier_std::Float64 = @addr(gamma(1, 1), :outlier_std)
    slope::Float64 = @addr(normal(0, 2), :slope)
    intercept::Float64 = @addr(normal(0, 2), :intercept)
    means::Vector{Float64} = broadcast(+, slope * xs, intercept)
    ys::PersistentVector{Float64} = @addr(
        data_collapsed(means, fill(inlier_std, n), fill(outlier_std, n)), :data)
    return ys
end


#######################
# inference operators #
#######################

@compiled @gen function slope_proposal(prev)
    slope::Float64 = get_choices(prev)[:slope]
    @addr(normal(slope, 0.5), :slope)
end

@compiled @gen function intercept_proposal(prev)
    intercept::Float64 = get_choices(prev)[:intercept]
    @addr(normal(intercept, 0.5), :intercept)
end

@compiled @gen function inlier_std_proposal(prev)
    inlier_std::Float64 = get_choices(prev)[:inlier_std]
    @addr(normal(inlier_std, 0.5), :inlier_std)
end

@compiled @gen function outlier_std_proposal(prev)
    outlier_std::Float64 = get_choices(prev)[:outlier_std]
    @addr(normal(outlier_std, 0.5), :outlier_std)
end

@compiled @gen function flip_z(prob::Float64)
    @addr(bernoulli(prob), :z)
end

data_proposal = at_dynamic(flip_z, Int)

function logsumexp(arr)
    min_arr = maximum(arr)
    min_arr + log(sum(exp.(arr .- min_arr)))
end

@compiled @gen function is_outlier_proposal(prev, i::Int)
	prev_z::Bool = get_choices(prev)[:data => i => :z]
    @addr(data_proposal(i, (prev_z ? 0.0 : 1.0,)), :data)
end

@compiled @gen function observe_datum(y::Float64)
    @addr(dirac(y), :y)
end

observe_data = plate(observe_datum)

@compiled @gen function observer(ys::Vector{Float64})
    @addr(observe_data(ys), :data)
end

@gen function observer_collapsed(ys::Vector{Float64})
    for (i, y) in enumerate(ys)
        @addr(dirac(y), :data => i)
    end
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
xs = collect(range(-5, stop=5, length=1000))
ys = Float64[]
for (i, x) in enumerate(xs)
    if rand() < prob_outlier
        y = true_slope * x + true_intercept + randn() * true_inlier_noise
    else
        y = true_slope * x + true_intercept + randn() * true_outlier_noise
    end
    push!(ys, y)
end

######################
# inference programs #
######################

function do_inference_collapsed(n)
    observations = get_choices(simulate(observer_collapsed, (ys,)))

    # initial trace
    (trace, weight) = generate(model_collapsed, (xs,), observations)

    for i=1:n
    
        # steps on the parameters
        for j=1:5
            trace = mh(model_collapsed, slope_proposal, (), trace)
            trace = mh(model_collapsed, intercept_proposal, (), trace)
            trace = mh(model_collapsed, inlier_std_proposal, (), trace)
            trace = mh(model_collapsed, outlier_std_proposal, (), trace)
        end
		choices = get_choices(trace)
		println((choices[:inlier_std], choices[:outlier_std], choices[:slope], choices[:intercept]))
    end

	choices = get_choices(trace)
    return (choices[:inlier_std], choices[:outlier_std], choices[:slope], choices[:intercept])
end

function do_inference(n)
    observations = get_choices(simulate(observer, (ys,)))

    # initial trace
    (trace, weight) = generate(model, (xs,), observations)

    for i=1:n
    
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
		choices = get_choices(trace)
		println((choices[:inlier_std], choices[:outlier_std], choices[:slope], choices[:intercept]))
    end

    choices = get_choices(trace)
    return (choices[:inlier_std], choices[:outlier_std], choices[:slope], choices[:intercept])
end


#################
# run inference #
#################

using Test

(inlier_std, outlier_std, slope, intercept) = do_inference(100)
max_std = max(inlier_std, outlier_std)
min_std = min(inlier_std, outlier_std)
@test isapprox(min_std, 0.5, atol=1e-1)
@test isapprox(max_std, 5.0, atol=5e-1)
@test isapprox(slope, -1, atol=1e-1)
@test isapprox(intercept, 2, atol=1e-1)

(inlier_std, outlier_std, slope, intercept) = do_inference_collapsed(100)
max_std = max(inlier_std, outlier_std)
min_std = min(inlier_std, outlier_std)
@test isapprox(min_std, 0.5, atol=1e-1)
@test isapprox(max_std, 5.0, atol=5e-1)
@test isapprox(slope, -1, atol=1e-1)
@test isapprox(intercept, 2, atol=1e-1)
