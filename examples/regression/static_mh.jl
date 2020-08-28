using Gen
import Random

include("static_model.jl")
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

Gen.load_generated_functions()

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
        println("score: $score, slope: $slope, intercept: $intercept, inlier_std: $inlier_std, outlier_std: $outlier_std")
    end
    return scores
end

(xs, ys) = make_data_set(200)
do_inference(xs, ys, 10)
@time do_inference(xs, ys, 50)

# handcoded version

mutable struct State
    xs::Vector{Float64}
    log_inlier_std::Float64
    log_outlier_std::Float64
    slope::Float64
    intercept::Float64
    is_outlier::Vector{Bool}
    ys::Vector{Float64}
end

function init_state(xs, ys)
    log_inlier_std = random(normal, 0, 2)
    log_outlier_std = random(normal, 0, 2)
    slope = random(normal, 0, 2)
    intercept = random(normal, 0, 2)
    is_outlier = Vector{Bool}(undef, length(xs))
    for i=1:length(xs)
        is_outlier[i] = (rand() < 0.5)
    end
    State(xs, log_inlier_std, log_outlier_std, slope, intercept, is_outlier, ys)
end

function log_likelihood(x, y, is_outlier, log_inlier_std, log_outlier_std, slope, intercept)
    std = is_outlier ? exp(log_outlier_std) : exp(log_inlier_std)
    logpdf(normal, y, slope * x + intercept, std)
end

function log_likelihood(state)
    ll = 0.
    for (x, is_outlier, y) in zip(state.xs, state.is_outlier, state.ys)
        ll += log_likelihood(x, y, is_outlier,
                    state.log_inlier_std, state.log_outlier_std, state.slope, state.intercept)
    end
    ll
end

macro parameter_mh_move!(state, addr)
    quote
        prev_ll = log_likelihood($(esc(state)))
        prev_val = $(esc(state)).$addr
        new_val = normal(prev_val, 0.5)
        prev_prior = logpdf(normal, prev_val, 0, 2)
        new_prior = logpdf(normal, new_val, 0, 2)
        $(esc(state)).$addr = new_val
        new_ll = log_likelihood($(esc(state)))
        if log(rand()) >= new_ll - prev_ll + new_prior - prev_prior
            $(esc(state)).$addr = prev_val # reject
        end
    end
end

function is_outlier_mh_move!(state, i::Int)
    prev_val = state.is_outlier[i]
    new_val = !prev_val
    x = state.xs[i]
    y = state.ys[i]
    prev_ll = log_likelihood(x, y, prev_val,
        state.log_inlier_std, state.log_outlier_std, state.slope, state.intercept)
    new_ll = log_likelihood(x, y, new_val,
        state.log_inlier_std, state.log_outlier_std, state.slope, state.intercept)
    if log(rand()) < new_ll - prev_ll
        state.is_outlier[i] = new_val
    end
end

function handcoded_inference(xs, ys, iters)
    state = init_state(xs, ys)
    for iter=1:iters
        for j=1:5
            @parameter_mh_move!(state, slope)
            @parameter_mh_move!(state, intercept)
            @parameter_mh_move!(state, log_inlier_std)
            @parameter_mh_move!(state, log_outlier_std)
        end
        for j=1:length(xs)
            is_outlier_mh_move!(state, j)
        end
    end
    println("slope: $(state.slope)")
    println("intercept: $(state.intercept)")
end

handcoded_inference(xs, ys, 50)
@time handcoded_inference(xs, ys, 50)
