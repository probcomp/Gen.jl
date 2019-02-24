using Gen
using PyPlot

include("model.jl")

const xlim = (-6, 6)
const ylim = (-6, 6)

function render(trace)

    xs, = get_args(trace)
    ys = [trace[i => :y] for i=1:length(xs)]
    degree = trace[:degree]
    coeffs = [trace[(:c, i)] for i=1:degree+1]
    curve_xs = collect(range(xlim[1], stop=xlim[2], length=300))
    curve_ys = [coeffs' * [x^i for i=0:length(coeffs)-1] for x in curve_xs]
    is_outliers = [trace[i => :is_outlier] for i=1:length(xs)]
    colors = [is_outlier ? "red" : "blue" for is_outlier in is_outliers]
    scatter(xs, ys, c=colors)
    plot(curve_xs, curve_ys, color="black")
    noise = trace[:noise]
    fill_between(curve_xs, curve_ys .- noise, curve_ys .+ (2 * noise), color="black", alpha=0.3)
    ax = gca()
    ax[:set_xlim](xlim)
    ax[:set_ylim](ylim)
end

function least_squares(xs, ys, degree)
    n = length(xs)
    X = hcat([[x^i for x in xs] for i=0:degree]...)
    @assert size(X) == (n, degree+1)
    coeffs = X \ ys
    @assert size(coeffs) == (degree+1,)
    return coeffs
end

import Random

function gibbs_discrete_probs(trace, addr, values)
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    weights = Vector{Float64}(undef, length(values))
    for (i, value) in enumerate(values)
        (_, weights[i]) = update(trace, args, argdiffs, choicemap((addr, value)))
    end
    exp.(weights .- logsumexp(weights))
end

@gen function custom_block_proposal(trace)
    xs, = get_args(trace)
    ys = get_retval(trace)
    n = length(xs)

    # sample new degree and noise from the prior
    new_degree = @trace(uniform_discrete(1, length(degree_prior)), :degree)
    new_noise = @trace(gamma(2, 2), :noise)

    # do least squares fit for new degree
    new_coeffs_fit = least_squares(xs, ys, new_degree)

    # add a small amount of noise around each parameter
    for i=1:new_degree+1
        @trace(normal(new_coeffs_fit[i], 0.1), (:c, i))
    end

    # obtain the trace after making this change
    constraints = choicemap((:noise, new_noise), (:degree, new_degree))
    for (i, new_coeff) in enumerate(new_coeffs_fit)
        constraints[(:c, i)] = new_coeff
    end
    new_trace, = update(trace, (xs,), (NoChange(),), constraints)

    # propose each is_outlier from its conditional distribution
    for i=1:n
        probs = gibbs_discrete_probs(new_trace, i => :is_outlier, [true, false])
        @trace(bernoulli(probs[1]), i => :is_outlier)
    end
end

@gen function coeff_random_walk_proposal(trace, i::Int)
    prev_coeff = trace[(:c, i)]
    @trace(normal(prev_coeff, 0.1), (:c, i))
end

@gen function flip_outlier_proposal(trace, i::Int)
    prev_is_outlier = trace[i => :is_outlier]
    @trace(bernoulli(prev_is_outlier ? 0.1 : 0.9), i => :is_outlier)
end

function fancy_inference(xs, ys, iters::Int)

    # obtain initial trace
    constraints = choicemap()
    for (i, y) in enumerate(ys)
        constraints[i => :y] = y
    end
    trace, = generate(model, (xs,), constraints)

    traces = Vector{Any}(undef, iters)
    degree_accepted = Vector{Bool}(undef, iters)

    # do MCMC
    for iter=1:iters

        traces[iter] = trace

        # do a global move on the degree and all coefficients
        trace, _ = mh(trace, select(:degree))
        trace, degree_accepted[iter] = mh(trace, custom_block_proposal, ())

        # random walk moves on the coefficients
        degree = trace[:degree]
        for i=1:degree+1
            trace, _ = mh(trace, select((:c, i)))
        end

        # resimulation moves
        trace, _ = mh(trace, select(:noise))

        # sweep through all outlier variables
        for i=1:length(xs)
            trace, _ = mh(trace, select(i => :is_outlier))
        end
    end

    traces, degree_accepted
end

function basic_inference(xs, ys, iters::Int)

    # obtain initial trace
    constraints = choicemap()
    for (i, y) in enumerate(ys)
        constraints[i => :y] = y
    end
    trace, = generate(model, (xs,), constraints)

    traces = Vector{Any}(undef, iters)
    degree_accepted = Vector{Bool}(undef, iters)

    # do MCMC
    for iter=1:iters

        traces[iter] = trace

        # do a global move on the degree
        trace, degree_accepted[iter]= mh(trace, select(:degree))

        # random walk moves on the coefficients
        degree = trace[:degree]
        for i=1:degree+1
            trace, _ = mh(trace, select((:c, i)))
        end

        # resimulation moves
        trace, _ = mh(trace, select(:noise))

        # sweep through all outlier variables
        for i=1:length(xs)
            trace, _ = mh(trace, select(i => :is_outlier))
        end
    end

    traces, degree_accepted
end

function map_optimize_is_outlier(trace, i::Int)
    args = get_args(trace)
    argdiffs = ((NoChange() for _ in args)...,)
    (trace1, w1, _, _) = update(trace, args, argdiffs, choicemap((i => :is_outlier, false)))
    (trace2, w2, _, _) = update(trace, args, argdiffs, choicemap((i => :is_outlier, true)))
    if w2 > w1
        return trace2
    else
        return trace1
    end
end

function map_optimize_discrete(trace, addr, values)
    args = get_args(trace)
    argdiffs = ((NoChange() for _ in args)...,)
    weights = Vector{Float64}(undef, length(domain))
    traces = Vector{Any}(undef, length(domain))
    for (i, val) in enumerate(values)
        (traces[i], weights[i], _, _) = update(trace, args, argdiffs, choicemap((addr, val)))
    end
    idx = argmax(weights)
    return traces[idx]
end

function map_inference(xs, ys, iters::Int)

    # obtain initial trace
    constraints = choicemap()
    for (i, y) in enumerate(ys)
        constraints[i => :y] = y
    end
    constraints[:degree] = length(degree_prior) # maximum degree.
    trace, = generate(model, (xs,), constraints)

    traces = Vector{Any}(undef, iters)

    # do MCMC
    for iter=1:iters

        traces[iter] = trace

        # map optimization over the coefficients
        degree = trace[:degree]
        trace = map_optimize(trace, select([(:c, i) for i=1:degree+1]...))
        #trace, = mala(trace, select([(:c, i) for i=1:degree+1]...), 0.1)

        # resimulation move
        #trace, _ = mh(trace, select(:prob_outlier))
        trace, _ = mh(trace, select(:noise))

        # sweep through all outlier variables
        for i=1:length(xs)
            #trace = map_optimize_is_outlier(trace, i)
            trace = map_optimize_discrete(trace, i => :is_outlier, [false, true])
            #trace, = mh(trace, select(i => :is_outlier))
        end
    end

    traces
end


function plot_traces(traces, degree_accepted)
    iters = length(traces)
    degrees = [tr[:degree] for tr in traces]
    plot(degrees, color="black")
    return
   
    coeffs = fill(NaN, (iters, length(degree_prior)+1))
    for (iter, tr) in enumerate(traces)
        for i=1:tr[:degree]+1
            coeffs[iter, i] = tr[(:c, i)]
        end
    end
    for i=1:length(degree_prior)+1
        plot(coeffs[:,i])
    end
    for (iter, accepted) in enumerate(degree_accepted)
        if accepted && (iter == 1 || traces[iter][:degree] != traces[iter-1][:degree])
            plot([iter, iter], [ylim[1], ylim[2]], color="black")
        end
    end
end

import Random
Random.seed!(1)

# synthetic data set
xs = collect(range(-5, stop=5, length=20))
#ys = 2 .+ (-1 * xs) .+ randn(length(xs)) * 0.5
ys = 0.1 * xs .^ 3 .- xs .+ randn(length(xs))
#xs = collect(range(-5, stop=5, length=20))
#ys = 2 .+ (0.5 .* (xs .^ 2)) .+ (-1 * xs) .+ randn(length(xs)) * 0.5
#ys[end] = 5
#ys[end-1] = 4
#ys[1] = -5
#ys[2] = -4

# held out test set (of inliers)
xs_test = collect(range(-5, stop=5, length=100))
ys_test = -xs_test .+ 2 .+ randn(length(xs_test)) * 0.5

# show samples from the prior
function show_prior_samples()
    figure(figsize=(12,12))
    for i=1:16
        subplot(4, 4, i)
        trace = simulate(model, (collect(range(-5, stop=5, length=30)),))
        render(trace)
    end
    tight_layout()
    savefig("prior_samples.png")
end
#show_prior_samples()

function held_out_mse(trace)
    degree = trace[:degree]
    coeffs = [trace[(:c, i)] for i=1:degree+1]
    ys_test_mean = [coeffs' * [x^i for i=0:length(coeffs)-1] for x in xs_test]
    diffs = ys_test_mean .- ys_test
    sqrt((diffs' * diffs) / length(xs_test))
end
    
# do inference
function show_inference_results_fancy()
    figure(figsize=(12,24))
    for i=1:16
        println("inference replicate $i..")
        traces, degree_accepted = fancy_inference(xs, ys, 200)
        subplot(8, 4, i)
        render(traces[end])
        subplot(8, 4, i + 16)
        plot_traces(traces, degree_accepted)
        #plot(1:length(traces), map(held_out_mse, traces))
        gca()[:set_ylim]((0, 10))
    end
    tight_layout()
    #savefig("inference_results_fancy.png")
    savefig("inference_results_fancy_full.png")

    #figure(figsize=(4,4))
    #for i=1:32
        #traces, degree_accepted = fancy_inference(xs, ys, 100)
        #plot(1:length(traces), map(held_out_mse, traces), alpha=0.5, color="black")
        #gca()[:set_ylim]((0, 10))
    #end
    #savefig("inference_results_fancy_composite_full.png")

    figure(figsize=(3,1.5))
    final_traces = []
    for i=1:50
        traces, degree_accepted = fancy_inference(xs, ys, 200)
        push!(final_traces, traces[end])
    end
    hist([tr[:degree] for tr in final_traces], bins=collect(0.5:1:5.5), align="mid")
    tight_layout()
    savefig("degree_histogram_fancy_200.pdf")
end
show_inference_results_fancy()

# do inference
function show_inference_results_basic()
    figure(figsize=(12,24))
    for i=1:16
        println("inference replicate $i..")
        traces, degree_accepted = basic_inference(xs, ys, 200)
        subplot(8, 4, i)
        render(traces[end])
        subplot(8, 4, i + 16)
        plot_traces(traces, degree_accepted)
        #plot(1:length(traces), map(held_out_mse, traces))
        gca()[:set_ylim]((0, 10))
    end
    tight_layout()
    savefig("inference_results_basic_full.png")

    #figure(figsize=(4,4))
    #for i=1:32
        #traces, degree_accepted = basic_inference(xs, ys, 100)
        #plot(1:length(traces), map(held_out_mse, traces), alpha=0.5, color="black")
        #gca()[:set_ylim]((0, 10))
    #end
    #savefig("inference_results_basic_composite_full.png")

    figure(figsize=(3,1.5))
    final_traces = []
    for i=1:50
        traces, degree_accepted = basic_inference(xs, ys, 200)
        push!(final_traces, traces[end])
    end
    hist([tr[:degree] for tr in final_traces], bins=collect(0.5:1:5.5), align="mid")
    tight_layout()
    savefig("degree_histogram_basic_200.pdf")
end
show_inference_results_basic()

# do high degree map inference
function show_map_inference_results()
    figure(figsize=(12,24))
    for i=1:16
        println("inference replicate $i..")
        traces = map_inference(xs, ys, 50)
        subplot(8, 4, i)
        render(traces[end])
        subplot(8, 4, i + 16)
        plot_traces(traces, [false for _ in traces])
        #plot(1:length(traces), map(held_out_mse, traces))
        gca()[:set_ylim]((0, 10))
    end
    tight_layout()
    savefig("inference_results_map.png")
end
#show_map_inference_results()
