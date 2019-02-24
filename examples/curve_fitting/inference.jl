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

@gen function custom_least_squares_proposal(trace)
    xs, = get_args(trace)
    ys = get_retval(trace)
    n = length(xs)
    inliers = [!trace[i => :is_outlier] for i=1:n]
    xs_inliers = xs[inliers]
    ys_inliers = ys[inliers]

    # sample new degree
    new_degree = @trace(uniform_discrete(1, length(degree_prior)), :degree)

    # do least squares fit for new degree
    new_coeffs_fit = least_squares(xs_inliers, ys_inliers, new_degree)

    # add noise around each parameter
    for i=1:new_degree+1
        @trace(normal(new_coeffs_fit[i], 0.3), (:c, i))
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
        trace, degree_accepted[iter]= mh(trace, custom_least_squares_proposal, ())

        # random walk moves on the coefficients
        degree = trace[:degree]
        for i=1:degree+1
            trace, _ = mh(trace, coeff_random_walk_proposal, (i,))
        end

        # resimulation moves
        trace, _ = mh(trace, select(:prob_outlier))
        trace, _ = mh(trace, select(:noise))

        # sweep through all outlier variables
        for i=1:length(xs)
            trace, _ = mh(trace, flip_outlier_proposal, (i,))
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
            trace, _ = mh(trace, coeff_random_walk_proposal, (i,))
        end

        # resimulation moves
        trace, _ = mh(trace, select(:prob_outlier))
        trace, _ = mh(trace, select(:noise))

        # sweep through all outlier variables
        for i=1:length(xs)
            trace, _ = mh(trace, flip_outlier_proposal, (i,))
        end
    end

    traces, degree_accepted
end

function high_degree_map_inference(xs, ys, iters::Int)

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

        # random walk moves on the coefficients
        degree = trace[:degree]
        trace = map_optimize(trace, select([(:c, i) for i=1:degree+1]...))

        # resimulation moves
        trace, _ = mh(trace, select(:prob_outlier))
        trace, _ = mh(trace, select(:noise))

        # sweep through all outlier variables
        for i=1:length(xs)
            trace, _ = mh(trace, flip_outlier_proposal, (i,))
        end
    end

    traces
end


function plot_traces(traces, degree_accepted)
    iters = length(traces)
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

# show samples from the prior
function show_prior_samples()
    figure(figsize=(12,12))
    for i=1:16
        subplot(4, 4, i)
        trace = simulate(model, (xs,))
        render(trace)
    end
    tight_layout()
    savefig("prior_samples.png")
end
#show_prior_samples()

# synthetic data set
xs = collect(range(-5, stop=5, length=10))
ys = -xs .+ 2 .+ randn(10) * 0.5
ys[end] = 5
ys[1] = -5
    
# do inference
function show_inference_results_fancy()
    figure(figsize=(12,24))
    for i=1:16
        println("inference replicate $i..")
        traces, degree_accepted = fancy_inference(xs, ys, 50)
        subplot(8, 4, i)
        render(traces[end])
        subplot(8, 4, i + 16)
        plot_traces(traces, degree_accepted)
    end
    tight_layout()
    savefig("inference_results_fancy.png")
end
#show_inference_results_fancy()

# do inference
function show_inference_results_basic()
    figure(figsize=(12,24))
    for i=1:16
        println("inference replicate $i..")
        traces, degree_accepted = basic_inference(xs, ys, 50)
        subplot(8, 4, i)
        render(traces[end])
        subplot(8, 4, i + 16)
        plot_traces(traces, degree_accepted)
    end
    tight_layout()
    savefig("inference_results_basic.png")
end
#show_inference_results_basic()

# do high degree map inference
function show_map_inference_results()
    figure(figsize=(12,24))
    for i=1:16
        println("inference replicate $i..")
        traces = high_degree_map_inference(xs, ys, 50)
        subplot(8, 4, i)
        render(traces[end])
        subplot(8, 4, i + 16)
        plot_traces(traces, [false for _ in traces])
    end
    tight_layout()
    savefig("inference_results_high_degree_map.png")
end
show_map_inference_results()
