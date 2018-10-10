using Gen
using PyPlot

include("univariate_gp.jl")


#############################
# custom change propagation #
#############################

struct ExtendInputs end

function compute_gp_arg_change(length_scale_change, noise1_change, noise2_change, args_change)

    # generate case (to get initial trace)
    if (args_change === nothing
        && length_scale_change === nothing
        && noise1_change === nothing
        && noise2_change === nothing)
        # Unknown
        return nothing

    # force_update case (during MCMC on hyperparameters)
    elseif (args_change == NoChange()
            && (isa(length_scale_change, Tuple) || length_scale_change == NoChange())
            && (isa(noise1_change, Tuple) || noise1_change == NoChange())
            && (isa(noise2_change, Tuple) || noise2_change == NoChange()))
        # just changing the GP parameters, not input
        return GPArgChangeInfo(true, false)

    # prediction on new data
    elseif (args_change == ExtendInputs()
            && length_scale_change == NoChange()
            && noise1_change == NoChange()
            && noise2_change == NoChange())
        # just changing the input, not the GP parameters
        return GPArgChangeInfo(false, true)
    else
        error("Not implemented")
    end
end


####################
# generative model #
####################

@gen function model(xs::Vector{Float64})

    # hyperparameters
    noise1 = @addr(gamma(1, 1), :noise1) + 0.001
    noise2 = @addr(gamma(1, 1), :noise2) + 0.001
    length_scale = @addr(gamma(1, 1), :length_scale)

    # mean and covariance functions
    mean_function = (x::Float64) -> 0.
    function cov_function(x1::Float64, x2::Float64)
        noise2 * exp((-0.5 / length_scale) * (x1 - x2) * (x1 - x2))
    end

    # sample output points from the GP
    gp_arg_change = compute_gp_arg_change(
        @change(:length_scale), @change(:noise1), @change(:noise2), @argschange())

    # invoke GP
    ys = @addr(gaussian_process(mean_function, cov_function, noise1, xs), :gp_outputs, gp_arg_change)

    return ys
end

######################################
# sample some datsets from the prior #
######################################

import Random
Random.seed!(1)
num_datasets = 25
num_datapoints = 100
xs_prior = collect(range(-5, stop=5, length=num_datapoints))
figure(figsize=(16,16))
for i=1:num_datasets
    trace =  simulate(model, (xs_prior,))
    assignment = get_assignment(trace)
    ys_prior = Float64[assignment[:gp_outputs => i] for i=1:num_datapoints]
    subplot(5, 5, i)
    scatter(xs_prior, ys_prior)
    gca()[:set_xlim]((-5, 5))
    gca()[:set_ylim]((-5, 5))
end
savefig("prior_samples.png")


########################
# inference experiment #
########################

# observe first 50 data points
import CSV
df = CSV.read("01-airline.processed.txt", header=[:x, :y])
xs_init = Vector{Float64}(df[:x])
ys_init = Vector{Float64}(df[:y])
ys_init = (ys_init .- mean(ys_init)) ./ std(ys_init)
num_init = length(xs_init)
#data = CSV.read(
#xs_init = collect(range(-5, stop=0, length=50))
#ys_init = sin.(0.1 * xs_init)
#ys_init = xs_init + randn(length(xs_init)) * 0.1
constraints = DynamicAssignment()
for (i, y) in enumerate(ys_init)
    constraints[:gp_outputs => i] = y
end
(trace, _) =  generate(model, (xs_init,), constraints)

# do inference over length scale
@gen function length_scale_proposal(prev_trace)
    @addr(gamma(1, 1), :length_scale)    
end

@gen function noise1_proposal(prev_trace)
    @addr(gamma(1, 1), :noise1)    
end

@gen function noise2_proposal(prev_trace)
    @addr(gamma(1, 1), :noise2)
end

tic()
println("mcmc...")
for mcmc_iteration=1:10000
    trace = mh(model, length_scale_proposal, (), trace)
    trace = mh(model, noise1_proposal, (), trace)
    trace = mh(model, noise2_proposal, (), trace)
end
toc()

# predict next 50 data points
tic()
println("prediction...")
new_xs = collect(range(12, stop=20, length=50))
#new_xs = collect(range(0, stop=5, length=50))
xs = vcat(xs_init, new_xs)
colors = vcat(["blue" for _=1:num_init], ["red" for _=1:50])
num_predictions = 25
figure(figsize=(16,16))
for i=1:num_predictions
    new_trace = predict(model, (xs,), ExtendInputs(), trace)
    assignment = get_assignment(new_trace)
    ys = Float64[assignment[:gp_outputs => i] for i=1:length(xs)]
    subplot(5, 5, i)
    scatter(xs, ys, c=colors)
    gca()[:set_xlim]((0, 20))
    gca()[:set_ylim]((
        minimum(ys_init) - 0.1 * (maximum(ys_init) - minimum(ys_init)),
        maximum(ys_init) + 0.1 * (maximum(ys_init) - minimum(ys_init))))
end
savefig("predictions.png")
toc()
