using Gen
import Random

#########
# model #
#########

struct Params
    prob_outlier::Float64
    inlier_std::Float64
    outlier_std::Float64
    slope::Float64
    intercept::Float64
end

@gen function datum(x::Float64, params::Params)
    is_outlier = @addr(bernoulli(params.prob_outlier), :z)
    std = is_outlier ? params.inlier_std : params.outlier_std
    y = @addr(normal(x * params.slope + params.intercept, std), :y)
    return y
end

@gen function model(xs::Vector{Float64})
    inlier_std = @addr(Gen.gamma(1, 1), :inlier_std)
    outlier_std = @addr(Gen.gamma(1, 1), :outlier_std)
    slope = @addr(normal(0, 2), :slope)
    intercept = @addr(normal(0, 2), :intercept)
    params = Params(0.5, inlier_std, outlier_std, slope, intercept)
    ys = Vector{Float64}(undef, length(xs))
    for (i, x) in enumerate(xs)
        ys[i] = @addr(datum(x, params), :data => i)
    end
    return ys
end

#######################
# inference operators #
#######################

@sel function slope_sel() @select(:slope) end
@sel function intercept_sel() @select(:intercept) end
@sel function inlier_std_sel() @select(:inlier_std) end
@sel function outlier_std_sel() @select(:outlier_std) end

@sel function params()
    @select(:slope)
    @select(:intercept)
end

@sel function variances()
    @select(:inlier_std)
    @select(:outlier_std)
end

@gen function is_outlier_proposal(prev, i::Int)
    prev = get_assignment(prev)[:data => i => :z]
    @addr(bernoulli(prev ? 0.0 : 1.0), :data => i => :z)
end

@gen function observer(ys::Vector{Float64})
    for (i, y) in enumerate(ys)
        @addr(dirac(y), :data => i => :y)
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


function do_inference(n)
    observations = get_assignment(simulate(observer, (ys,)))
    
    # initial trace
    (trace, _) = generate(model, (xs,), observations)
    
    for i=1:n
    
        # steps on the parameters
        for j=1:5
            trace = mh(model, slope_sel, (), trace)
            trace = mh(model, intercept_sel, (), trace)
            trace = mh(model, inlier_std_sel, (), trace)
            trace = mh(model, outlier_std_sel, (), trace)
        end
    
        # step on the outliers
        for j=1:length(xs)
            trace = mh(model, is_outlier_proposal, (j,), trace)
        end
    
        score = get_call_record(trace).score
    
        # print
        assignment = get_assignment(trace)
        slope = assignment[:slope]
        intercept = assignment[:intercept]
        inlier_std = assignment[:inlier_std]
        outlier_std = assignment[:outlier_std]
        println("score: $score, slope: $slope, intercept: $intercept, inlier_std: $inlier_std, outlier_std: $outlier_std")
    end
end

@time do_inference(100)
@time do_inference(100)
