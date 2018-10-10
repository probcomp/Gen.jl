using Gen
using PyPlot

include("univariate_gp.jl")

# cases we should handle:

# 1) generate
#       @argschange() = nothing
#       @change(:length_scale) = nothing
#       gp_arg_change = nothing

# 2) force_update (during MCMC on length_scale)
#       @argschange() = NoChange()
#       @change(:length_scale) = (true, prev_value)
#       gp_arg_change = GPArgChangeInfo(true, false)

# 3) predict (fix_update; extending x)
#       @argschange() = ExtendInputs()
#       @change(:length_scale) = NoChange()
#       gp_arg_change = GPArgChangeInfo(false, true)

struct ExtendInputs end

function compute_gp_arg_change(length_scale_change, noise1_change, noise2_change, args_change)
    if (args_change === nothing
        && length_scale_change === nothing
        && noise1_change === nothing
        && noise2_change === nothing)
        # Unknown
        return nothing
    elseif (args_change == NoChange()
            && isa(length_scale_change, Tuple)
            && isa(noise1_change, Tuple)
            && isa(noise2_change, Tuple))
        # just changing the GP parameters, not input
        return GPArgChangeInfo(true, false)
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
    ys = @addr(gaussian_process(mean_function, cov_function, noise1, xs), :gp_outputs, gp_arg_change)
    return ys
end

# sample some datasets from the model
import Random
Random.seed!(1)
num_datasets = 25
num_samples = 100
xs_prior = collect(range(-5, stop=5, length=num_samples))
figure(figsize=(16,16))
for i=1:num_datasets
    trace =  simulate(model, (xs_prior,))
    assignment = get_assignment(trace)
    ys_prior = Float64[assignment[:gp_outputs => i] for i=1:num_samples]
    subplot(5, 5, i)
    scatter(xs_prior, ys_prior)
    gca()[:set_xlim]((-5, 5))
    gca()[:set_ylim]((-5, 5))
end
savefig("prior_samples.png")


exit()

# observe first 50 data points
xs_init = collect(range(-5, stop=0, length=50))
ys_init = xs_init + randn(length(xs_init)) * 0.1
constraints = DynamicAssignment()
for (i, y) in enumerate(ys_init)
    constraints[:gp_outputs => i] = y
end
(trace, _) =  generate(model, (xs_init,), constraints)

# do inference over length scale
@gen function length_scale_proposal(prev_trace)
    @addr(gamma(1, 1), :length_scale)    
end
#for mcmc_iteration=1:100
    #trace = mh(model, length_scale_proposal, (), trace)
#end

# predict next 50 data points
new_xs = collect(range(0, stop=5, length=50))
xs = vcat(xs_init, new_xs)
new_trace = predict(model, (xs,), ExtendInputs(), trace)
assignment = get_assignment(new_trace)
ys = Float64[assignment[:gp_outputs => i] for i=1:length(xs)]

figure()
scatter(xs, ys)
savefig("predictions.png")
