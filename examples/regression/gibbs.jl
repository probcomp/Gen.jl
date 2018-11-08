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

data = plate(datum)

@gen function model(xs::Vector{Float64})
    inlier_std = @addr(gamma(1, 1), :inlier_std)
    outlier_std = @addr(gamma(1, 1), :outlier_std)
    slope = @addr(normal(0, 2), :slope)
    intercept = @addr(normal(0, 2), :intercept)
    params = Params(0.5, inlier_std, outlier_std, slope, intercept)
    @diff begin
        argdiff = noargdiff
        for addr in [:slope, :intercept, :inlier_std, :outlier_std]
            if !isnodiff(@choicediff(addr))
                argdiff = unknownargdiff
            end
        end
    end
    ys = @addr(data(xs, fill(params, length(xs))), :data, argdiff)
    return ys
end

#######################
# inference operators #
#######################

@gen function slope_proposal(prev)
    slope = get_assignment(prev)[:slope]
    @addr(normal(slope, 0.5), :slope)
end

@gen function intercept_proposal(prev)
    intercept = get_assignment(prev)[:intercept]
    @addr(normal(intercept, 0.5), :intercept)
end

@gen function inlier_std_proposal(prev)
    inlier_std = get_assignment(prev)[:inlier_std]
    @addr(normal(inlier_std, 0.5), :inlier_std)
end

@gen function outlier_std_proposal(prev)
    outlier_std = get_assignment(prev)[:outlier_std]
    @addr(normal(outlier_std, 0.5), :outlier_std)
end

function logsumexp(arr)
    min_arr = maximum(arr)
    min_arr + log(sum(exp.(arr .- min_arr)))
end

@gen function is_outlier_proposal(prev, i::Int)
    prev_args = get_call_record(prev).args
    constraints = DynamicAssignment()
    constraints[:data => i => :z] = true
    (tmp1,) =  update(model, prev_args, noargdiff, prev, constraints)
    constraints[:data => i => :z] = false
    (tmp2,) = update(model, prev_args, noargdiff, prev, constraints)
    log_prob_true = get_call_record(tmp1).score 
    log_prob_false = get_call_record(tmp2).score
    prob_true = exp(log_prob_true - logsumexp([log_prob_true, log_prob_false]))
    @addr(bernoulli(prob_true), :data => i => :z)
end

@gen function is_outlier_proposal2(prev, i::Int)
    assignment = get_assignment(prev)
    slope = assignment[:slope]
    intercept = assignment[:intercept]
    inlier_std = assignment[:inlier_std]
    outlier_std = assignment[:outlier_std]
    x = get_call_record(prev).args[1][i]
    mu = intercept + slope * x
    log_prob_true = logpdf(normal, assignment[:data => i => :y], mu, outlier_std)
    log_prob_false = logpdf(normal, assignment[:data => i => :y], mu, inlier_std)
    prob_true = exp(log_prob_true - logsumexp([log_prob_true, log_prob_false]))
    @addr(bernoulli(prob_true), :data => i => :z)
end

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

    observations = DynamicAssignment()
    for (i, y) in enumerate(ys)
        observations[:data => i => :y] = y
    end
    
    # initial trace
    (trace, _) = generate(model, (xs,), observations)
    
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
            trace = mh(model, is_outlier_proposal2, (j,), trace)
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
