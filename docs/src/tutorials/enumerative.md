# [Debugging Models with Enumerative Inference](@id enumerative_tutorial)

When working with probabilistic models, we often rely on Monte Carlo methods like importance sampling (as introduced in [Introduction to Modeling](@ref modeling_tutorial)) and MCMC (as introduced in [Basics of MCMC](@ref mcmc_map_tutorial)) to approximate posterior distributions. But how can we tell if these approximations are actually working correctly? Sometimes poor inference results stem from bugs in our inference algorithms, while other times they reveal fundamental issues with our model specification.

This tutorial introduces *enumerative inference* as a debugging tool. Unlike the sampling-based methods we've seen in previous tutorials, which draw samples that are approximately distributed according to the posterior distribution over latent values, enumerative inference systematically evaluates the posterior probability of every value in the latent space (or a discretized version of this space).

When the latent space isn't too large (e.g. not too many dimensions), this approach can compute a "gold standard" posterior approximation that other methods can be compared against, helping us distinguish between inference failures and model misspecification. Enumerative inference is often slower than a well-tuned Monte Carlo algorithm (since it may enumerate over regions with very low probability), but having a gold-standard posterior allows us to check that faster and more efficient algorithms are working correctly.

```@setup enumerative_tutorial
using Gen, Plots, StatsBase, Printf
import Random
Random.seed!(0)
```

## Enumeration for Discrete Models

Let's start with a simple example where enumeration can be used to perform *exact inference*, computing the exact posterior probability for every possible combination of discrete latent variables. We'll build a robust Bayesian linear regression model, but unlike the continuous model from the MCMC tutorial, we'll use discrete priors for all latent variables.

```@example enumerative_tutorial
@gen function discrete_regression(xs::Vector{<:Real})
    # Discrete priors for slope and intercept
    slope ~ uniform_discrete(-2, 2)  # Slopes: -2, -1, 0, 1, 2
    intercept ~ uniform_discrete(-2, 2)  # Intercepts: -2, -1, 0, 1, 2
    
    # Sample outlier classification and y value for each x value
    n = length(xs)
    ys = Float64[]
    for i = 1:n
        # Prior on outlier probability
        is_outlier = {:data => i => :is_outlier} ~ bernoulli(0.1)
        
        if is_outlier
            # Outliers have large noise
            y = {:data => i => :y} ~ normal(0., 5.)
        else
            # Inliers follow the linear relationship, with low noise
            y_mean = slope * xs[i] + intercept
            y = {:data => i => :y} ~ normal(y_mean, 1.)
        end
        push!(ys, y)
    end
    
    return ys
end
nothing # hide
```

Let's generate some synthetic data with a true slope of 1 and intercept of 0:

```@example enumerative_tutorial
# Generate synthetic data
true_slope = 1
true_intercept = 0
xs = [-2., -1., 0., 1., 2.]
ys = true_slope .* xs .+ true_intercept .+ 1.0 * randn(5)

# Make one point an outlier
ys[3] = 4.0

# Visualize the data
point_colors = [:blue, :blue, :red, :blue, :blue]
scatter(xs, ys, label="Observations", markersize=6, xlabel="x", ylabel="y",
        color=point_colors)
plot!(xs, true_slope .* xs .+ true_intercept, 
      label="True line", linestyle=:dash, linewidth=2, color=:black)
```

Now we can use enumerative inference to compute the exact posterior. We'll enumerate over all possible combinations of slope, intercept, and outlier classifications:

```@example enumerative_tutorial
# Create observations choicemap
observations = choicemap()
for (i, y) in enumerate(ys)
    observations[:data => i => :y] = y
end

# Set up the enumeration grid
# We enumerate over discrete slope, intercept, and outlier classifications
grid_specs = Tuple[
    (:slope, -2:2),  # 5 possible slopes
    (:intercept, -2:2),  # 3 possible intercepts
]
for i in 1:length(xs)
    push!(grid_specs, (:data => i => :is_outlier, [false, true]))
end

# Create the enumeration grid
grid = choice_vol_grid(grid_specs...)
nothing # hide
```

Here, we used [`choice_vol_grid`](@ref) to enumerate over all possible combinations of slope, intercept, and outlier classifications. The resulting `grid` object is a multi-dimensional iterator, where each element consists of a [`ChoiceMap`](@ref) that specifies the values of all latent variables, and the log-volume of latent space covered by that element of the grid. Since all latent variables are discrete, the volume of latent space covered by each element is equal to 1 (and hence the log-volume is 0). We can inspect the first element of this grid using the `first` function:

```@example enumerative_tutorial
choices, log_vol = first(grid)
println("Log volume: ", log_vol)
println("Choices: ")
show(stdout, "text/plain", choices)
```

Having constructed the enumeration grid, we now pass this to the [`enumerative_inference`](@ref) function, along with the generative model (`discrete_regression`), model arguments (in this case, `xs`), and the `observations`:

```@example enumerative_tutorial
# Run enumerative inference
traces, log_norm_weights, lml_est = 
    enumerative_inference(discrete_regression, (xs,), observations, grid)

println("Grid size: ", size(grid))
println("Log marginal likelihood: ", lml_est)
```

The [`enumerative_inference`](@ref) function returns an array of `traces` and an array of normalized log posterior probabilities (`log_norm_weights`) with the same shape as the input `grid`. It also returns an estimate of the log marginal likelihood (`lml_est`) of the observations. The estimate is *exact* in this case, since we enumerated over all possible combinations of latent variables.

Each trace corresponds to a full combination of the latent variables that were enumerated over. As such, the `log_norm_weights` array represents the *joint* posterior distribution over all latent variables. By summing over all traces which have the same value for a specific latent variable (or equivalently, by summing over a dimension of the `log_norm_weights` array), we can compute the *marginal* posterior distribution for that variable.  We'll do this below for the `slope` and `intercept` variables:

```@example enumerative_tutorial
# Compute 2D marginal posterior over slope and intercept
sum_dims = Tuple(3:ndims(log_norm_weights)) # Sum over all other variables
posterior_grid = sum(exp.(log_norm_weights), dims=sum_dims)
posterior_grid = dropdims(posterior_grid; dims=sum_dims)
```

Let's visualize the marginal posteriors over these variables, as well the joint posterior for both variables together. Below is some code to plot a 2D posterior heatmap with 1D marginals as histograms.

```@raw html
<details> <summary>Code to plot 2D grid of posterior values</summary>
```

```@example enumerative_tutorial
using Plots

function plot_posterior_grid(
    x_range, y_range, xy_probs;
    is_discrete = true, x_true = missing, y_true = missing,
    xlabel = "", ylabel = ""
)
    # Create the main heatmap
    p_main = heatmap(x_range, y_range, xy_probs, colorbar=false, widen=false,
                     color=:grays, xlabel=xlabel, ylabel=ylabel)
    if is_discrete
        # Add true parameters
        scatter!(p_main, [x_true], [y_true], legend=true,
                 markersize=36, markershape=:rect, color=:white,
                 markerstrokecolor=:red, label="True Parameters")
        # Annotate each cell with its posterior probability
        for idx in CartesianIndices(xy_probs)
            i, j = Tuple(idx)
            prob_str = @sprintf("%.3f", xy_probs[j, i])
            prob_color = xy_probs[j, i] > 0.2 ? :black : :white
            annotate!(x_range[i], y_range[j],
                      text(prob_str, color = prob_color, pointsize=12))
        end
    else
        # Add true parameters
        if !ismissing(x_true) && !ismissing(y_true)    
            scatter!(p_main, [x_true], [y_true], legend=true,
                     markersize=6, color=:red, markershape=:cross,
                     label="True Parameters")
        end
        if !ismissing(x_true)
            vline!([x_true], linestyle=:dash, linewidth=1, color=:red,
                   label="", alpha=0.5)
        end
        if !ismissing(y_true)
            hline!([y_true], linestyle=:dash, linewidth=1, color=:red,
                   label="", alpha=0.5)
        end
    end

    # Create 1D marginal histograms
    x_probs = vec(sum(xy_probs, dims=1))
    y_probs = vec(sum(xy_probs, dims=2))
    p_top = bar(x_range, x_probs, orientation=:v, ylims=(0, maximum(x_probs)),
                bar_width=diff(x_range)[1], linewidth=0, color=:black,
                showaxis=true, ticks=false, legend=false, widen=false)
    p_right = bar(y_range, y_probs, orientation=:h, xlims=(0, maximum(y_probs)),
                  bar_width=diff(y_range)[1], linewidth=0, color=:black,
                  showaxis=true, ticks=false, legend=false, widen=false)
    if !is_discrete
        xlims!(p_top, xlims(p_main))
        ylims!(p_right, ylims(p_main))
        if !ismissing(x_true)
            vline!(p_top, [x_true], linestyle=:dash,
                   linewidth=1, color=:red, legend=false)
        end
        if !ismissing(y_true)
            hline!(p_right, [y_true], linestyle=:dash,
                   linewidth=1, color=:red, legend=false)
        end
    end

    # Create empty plot for top-right corner
    p_empty = plot(legend=false, grid=false, showaxis=false, ticks=false)

    # Combine plots using layout
    plot(p_top, p_empty, p_main, p_right, 
         layout=@layout([a{0.9w,0.1h} b{0.1w,0.1h}; c{0.9w,0.9h} d{0.1w,0.9h}]),
         size=(750, 750))
end
nothing # hide
```

```@raw html
</details>
```

```@example enumerative_tutorial
# Extract parameter ranges
slope_range = [trs[1][:slope] for trs in eachslice(traces, dims=1)]
intercept_range = [trs[1][:intercept] for trs in eachslice(traces, dims=2)]

# Plot 2D posterior heatmap with 1D marginals as histograms
plot_posterior_grid(intercept_range, slope_range, posterior_grid,
                    x_true = true_intercept, y_true = true_slope,
                    xlabel = "Intercept", ylabel = "Slope",
                    is_discrete = true)
```

As can be seen, the posterior concentrates around the true values of the slope and intercept, though there is some uncertainty about both.

We can also examine which points are most likely to be outliers:

```@example enumerative_tutorial
# Compute posterior probability of each point being an outlier
outlier_probs = zeros(length(xs))
for (j, trace) in enumerate(traces)
    for i in 1:length(xs)
        if trace[:data => i => :is_outlier]
            outlier_probs[i] += exp(log_norm_weights[j])
        end
    end
end

bar(1:length(xs), outlier_probs, 
    xlabel="x", ylabel="P(outlier | data)",
    color=:black, ylim=(0, 1), legend=false)
```

Notice that enumerative inference correctly identifies that point 3 (which we made an outlier) has a high probability of being an outlier, while maintaining uncertainty about the exact classifications.

## Enumeration for Continuous Models

Many generative models of interest have continuous latent variables. While we can't enumerate over continuous spaces exactly, we can create a discrete approximation of a continuous target distribution by defining a grid. Let's extend our model to use continuous priors:

```@example enumerative_tutorial
@gen function continuous_regression(xs::Vector{<:Real})
    # Continuous slope and intercept priors
    slope ~ normal(0, 1)
    intercept ~ normal(0, 2)
    
    # Sample outlier classification and y value for each x value
    n = length(xs)
    ys = Float64[]
    for i = 1:n
        # Prior on outlier probability
        is_outlier = {:data => i => :is_outlier} ~ bernoulli(0.1)
        
        if is_outlier
            # Outliers have large noise
            y = {:data => i => :y} ~ normal(0., 5.)
        else
            # Inliers follow the linear relationship, with low noise
            y_mean = slope * xs[i] + intercept
            y = {:data => i => :y} ~ normal(y_mean, 1.)
        end
        push!(ys, y)
    end
    
    return ys
end
nothing # hide
```

We now construct a grid over the latent space using [`choice_vol_grid`](@ref). For continuous variables, we need to provide a range of grid points (including start and end points), and specify that the variable is `:continuous`:

```@example enumerative_tutorial
grid = choice_vol_grid(
    (:slope, -3:0.25:3, :continuous),  # 24 grid intervals
    (:intercept, -4:0.5:4, :continuous),  # 16 grid intervals
    # Still enumerate exactly over outlier classifications
    (:data => 1 => :is_outlier, [false, true]),
    (:data => 2 => :is_outlier, [false, true]),
    (:data => 3 => :is_outlier, [false, true]),
    (:data => 4 => :is_outlier, [false, true]),
    (:data => 5 => :is_outlier, [false, true]);
    anchor = :midpoint # Anchor evaluation point at midpoint of each interval
)

println("Grid size for continuous model: ", size(grid))
println("Number of grid elements: ", length(grid))
```

When some variables are specified as `:continuous`, the [`choice_vol_grid`](@ref) function automatically computes the log-volume of each grid cell. Inspecting the first element of the grid, we see that the log-volume is equal to `log(0.25 * 0.5) ≈ -2.0794`, since that grid cell is covers a volume of 0.25 * 0.5 = 0.125 of the slope-intercept latent space. We also see that the `slope` and `intercept` variables lie at the midpoint of this grid cell, since the `anchor` keyword argument was set to `:midpoint`:

```@example enumerative_tutorial
choices, log_vol = first(grid)
println("Log volume: ", log_vol)
println("Choices: ")
choices
```

Now let's generate some synthetic data to do inference on. We'll use ground-truth continuous parameters that don't lie exactly on the grid, in order to show that enumerative inference can still produce a reasonable approximation when the posterior is sufficiently smooth.

```@example enumerative_tutorial
# Generate synthetic data
true_slope = -1.21
true_intercept = 2.56
xs = [-2., -1., 0., 1., 2.]
ys = true_slope .* xs .+ true_intercept .+ 1.0 * randn(5)

# Make one point an outlier
ys[2] = 0.

# Create observations choicemap
observations = choicemap()
for (i, y) in enumerate(ys)
    observations[:data => i => :y] = y
end

# Visualize the data
point_colors = [:blue, :red, :blue, :blue, :blue]
scatter(xs, ys, label="Observations", markersize=6, xlabel="x", ylabel="y",
        color=point_colors)
plot!(xs, true_slope .* xs .+ true_intercept, 
      label="True line", linestyle=:dash, linewidth=2, color=:black)
```

As in the discrete case, we can use [`enumerative_inference`](@ref) to perform inference on the continuous model:

```@example enumerative_tutorial
# Run inference on the continuous model
traces, log_norm_weights, lml_est = 
    enumerative_inference(continuous_regression, (xs,), observations, grid)

println("Log marginal likelihood: ", lml_est)
```

Again, we can visualize the joint posterior over the `slope` and `intercept` variables with the help of some plotting code.

```@example enumerative_tutorial
# Compute marginal posterior over slope and intercept
sum_dims = Tuple(3:ndims(log_norm_weights)) # Sum over all other variables
posterior_grid = sum(exp.(log_norm_weights), dims=sum_dims)
posterior_grid = dropdims(posterior_grid; dims=sum_dims)

# Extract parameter ranges
slope_range = [trs[1][:slope] for trs in eachslice(traces, dims=1)]
intercept_range = [trs[1][:intercept] for trs in eachslice(traces, dims=2)]

# Plot 2D posterior heatmap with 1D marginals as histograms
p = plot_posterior_grid(intercept_range, slope_range, posterior_grid,
                    x_true = true_intercept, y_true = true_slope,
                    xlabel = "Intercept", ylabel = "Slope", is_discrete = false)
```

We can see that the true parameters lie in a cell with reasonably high posterior probability, though there is a fair amount of uncertainty due to bimodal nature of the posterior distribution. This manifests in the posterior over outlier classifications as well:

```@example enumerative_tutorial
# Compute posterior probability of each point being an outlier
outlier_probs = zeros(length(xs))
for i = 1:length(xs)
    for (j, trace) in enumerate(traces)
        if trace[:data => i => :is_outlier]
            outlier_probs[i] += exp(log_norm_weights[j])
        end
    end
end

# Plot posterior probability of each point being an outlier
bar(1:length(xs), outlier_probs, 
    xlabel="x", ylabel="P(outlier | data)",
    color=:black, ylim=(0, 1), legend=false)
```

The points at both `x=1` and `x=2` are inferred to be possible outliers, corresponding to each possible mode of the full posterior distribution. By extracting the slice of the `log_norm_weights` array that corresponds to `x=2` being an outlier (i.e., when `data => 2 => :is_outlier` is `true`), we can visualize the posterior distribution over the `slope` and `intercept` variables conditional on `x=2` being an outlier. As shown below, this conditional posterior is no longer bimodal, and concentrates more closely around the true parameters.

```@example enumerative_tutorial
# Extract slice of weights corresponding to x=2 being an outlier
cond_log_norm_weights = log_norm_weights[:,:,:,end:end,:,:,:]

# Compute marginal posterior over slope & intercept given that x=2 is an outlier
sum_dims = Tuple(3:ndims(cond_log_norm_weights)) # Sum over all other variables
posterior_grid = sum(exp.(cond_log_norm_weights), dims=sum_dims)
posterior_grid = dropdims(posterior_grid; dims=sum_dims)

# Extract parameter ranges
slope_range = [trs[1][:slope] for trs in eachslice(traces, dims=1)]
intercept_range = [trs[1][:intercept] for trs in eachslice(traces, dims=2)]

# Plot 2D posterior heatmap with 1D marginals as histograms
plot_posterior_grid(intercept_range, slope_range, posterior_grid,
                    x_true = true_intercept, y_true = true_slope,
                    xlabel = "Intercept", ylabel = "Slope", is_discrete = false)
```

Instead of extracting a slice of the full weight array, we could also have used [`choice_vol_grid`](@ref) to construct an enumeration grid with `data => 2 => :is_outlier` constrained to `true`, and then called [`enumerative_inference`](@ref) with this conditional grid. This ability to compute conditional posteriors is another useful aspect of enumerative inference: Even when the latent space becomes too high-dimensional for enumeration over the full joint posterior, we can still inspect the conditional posteriors over some variables conditioned on the values of other variables, and check whether they make sense.

## Diagnosing Model Misspecification

As we have seen above, enumerative inference allows us to approximate a posterior distribution with a high degree of fidelity (at the expense of additional computation). This allows us to distinguish between two ways that inference in a Bayesian model can go wrong:

- **Inference Failure:** The inference algorithm fails to approximate the true posterior distribution well (e.g. due to a bad importance sampling proposal, a poorly-designed MCMC kernel, or insufficient computation).

- **Model Misspecification:** The Bayesian model itself is misspecified, such that the true posterior distribution does not correspond with our intuitions about what the posterior should look like.

Both of these issues can occur at the same time: an algorithm might fail to converge to the true posterior, and the model might be misspecified. Regardless, since enumerative inference can approximate the true posterior distribution arbitrarily well (by making the grid arbitrarily large and fine), we can use it to check whether some other algorithm converges to the true posterior, and also whether the true posterior itself concords with our intuitions.

As a demonstration, let us write a version of the continuous regression model with narrow slope and intercept priors, and a high probability of outliers:

```@example enumerative_tutorial
@gen function misspecified_regression(xs::Vector{<:Real})
    # Narrow slope and intercept priors
    slope ~ normal(0, sqrt(0.5))
    intercept ~ normal(0, sqrt(0.5))
    
    # Sample outlier classification and y value for each x value
    n = length(xs)
    ys = Float64[]
    for i = 1:n
        # High (25% chance) prior probability of being an outlier
        is_outlier = {:data => i => :is_outlier} ~ bernoulli(0.25)
        
        if is_outlier
            # Outliers have large noise
            y = {:data => i => :y} ~ normal(0., 5.)
        else
            # Inliers follow the linear relationship, with low noise
            y_mean = slope * xs[i] + intercept
            y = {:data => i => :y} ~ normal(y_mean, 1.)
        end
        push!(ys, y)
    end
    
    return ys
end
nothing # hide
```

To create a case where the model is misspecified, we generate data with a steep slope and a large intercept, but no outliers:

```@example enumerative_tutorial
# Generate synthetic data
true_slope = 2.8
true_intercept = -2.4
xs = [-2., -1., 0., 1., 2.]
ys = true_slope .* xs .+ true_intercept .+ 1.0 * randn(5)

# Create observations choicemap
observations = choicemap()
for (i, y) in enumerate(ys)
    observations[:data => i => :y] = y
end

# Visualize the data
point_colors = [:blue, :blue, :blue, :blue, :blue]
scatter(xs, ys, label="Observations", markersize=6, xlabel="x", ylabel="y",
        color=point_colors)
plot!(xs, true_slope .* xs .+ true_intercept, 
      label="True line", linestyle=:dash, linewidth=2, color=:black)
```

Now let us try using [`importance_resampling`](@ref) to approximate the posterior distribution under the misspecified model:

```@example enumerative_tutorial
# Try importance resampling with 2000 inner samples and 100 outer samples
println("Running importance sampling...")
traces = [importance_resampling(misspecified_regression, (xs,), observations, 2000)[1] for i in 1:100]

# Compute the mean slope and intercept
mean_slope = sum(trace[:slope] for trace in traces) / length(traces)
mean_intercept = sum(trace[:intercept] for trace in traces) / length(traces)

println("Mean slope: ", mean_slope)
println("Mean intercept: ", mean_intercept)
```

Instead of recovering anything close to the true parameters, importance sampling infers a much smaller mean for the slope and intercept. We can also visualize the joint posterior over the slope and intercept by plotting a 2D histogram from the samples:

```@raw html
<details> <summary>Code to plot posterior samples</summary>
```

```@example enumerative_tutorial
function plot_posterior_samples(
    x_range, y_range, x_values, y_values;
    x_true = missing, y_true = missing,
    xlabel = "", ylabel = ""
)
    # Create the main heatmap
    p_main = histogram2d(x_values, y_values, bins=(x_range, y_range),
                         show_empty_bins=true, normalize=:probability,
                         color=:grays, colorbar=false, legend=false,
                         xlabel=xlabel, ylabel=ylabel)
    xlims!(p_main, minimum(x_range), maximum(x_range))
    ylims!(p_main, minimum(y_range), maximum(y_range))
    # Add true parameters
    if !ismissing(x_true) && !ismissing(y_true)    
        scatter!(p_main, [true_intercept], [true_slope], 
                 markersize=6, color=:red, markershape=:cross,
                 label="True Parameters", legend=true)
    end
    if !ismissing(x_true)
        vline!([x_true], linestyle=:dash, linewidth=1, color=:red,
                label="", alpha=0.5)
    end
    if !ismissing(y_true)
        hline!([y_true], linestyle=:dash, linewidth=1, color=:red,
                label="", alpha=0.5)
    end

    # Create 1D marginal histograms
    p_top = histogram(x_values, bins=x_range, orientation=:v, legend=false,
                      normalize=:probability, linewidth=0, color=:black,
                      showaxis=true, ticks=false)
    x_probs_max = maximum(p_top.series_list[2].plotattributes[:y])
    ylims!(p_top, 0, x_probs_max)
    xlims!(p_top, minimum(x_range), maximum(x_range))
    p_right = histogram(y_values, bins=y_range, orientation=:h, legend=false,
                        normalize=:probability, linewidth=0, color=:black,
                        showaxis=true, ticks=false)
    y_probs_max = maximum(p_right.series_list[2].plotattributes[:y])
    xlims!(p_right, 0, y_probs_max)
    ylims!(p_right, minimum(y_range), maximum(y_range))
    # Add true parameters
    if !ismissing(x_true)
        vline!(p_top, [x_true], linestyle=:dash,
               linewidth=1, color=:red, legend=false)
    end
    if !ismissing(y_true)
        hline!(p_right, [y_true], linestyle=:dash,
               linewidth=1, color=:red, legend=false)
    end

    # Create empty plot for top-right corner
    p_empty = plot(legend=false, grid=false, showaxis=false, ticks=false)

    # Combine plots using layout
    plot(p_top, p_empty, p_main, p_right, 
         layout=@layout([a{0.9w,0.1h} b{0.1w,0.1h}; c{0.9w,0.9h} d{0.1w,0.9h}]),
         size=(750, 750))
end
nothing # hide
```

```@raw html
</details>
```

```@example enumerative_tutorial
# Plot a 2D histogram for the slope and intercept variables
slopes = [trace[:slope] for trace in traces]
intercepts = [trace[:intercept] for trace in traces]
plot_posterior_samples(-4:0.25:4, -4:0.25:4, intercepts, slopes,
                       x_true=true_intercept, y_true=true_slope,
                       xlabel="Intercept", ylabel="Slope")
```

The distribution of samples produced by importance sampling lies far from the true slope and intercept, and concentrates around values that do not intuitively make sense given the data. The distribution over outlier classifications sheds some light on the problem:

```@example enumerative_tutorial
# Estimate posterior probability of each point being an outlier
outlier_probs = zeros(length(xs))
for i = 1:length(xs)
    for (j, trace) in enumerate(traces)
        if trace[:data => i => :is_outlier]
            outlier_probs[i] += 1/length(traces)
        end
    end
end

# Plot posterior probability of each point being an outlier
bar(1:length(xs), outlier_probs, 
    xlabel="x", ylabel="P(outlier | data)",
    color=:black, ylim=(0, 1), legend=false)
```

Importance sampling infers that many of the points are likely to be outliers. That is, instead of inferring a steep slope and a negative intercept, importance sampling prefers to explain the data as a flatter line with *many* outliers. 

These inferences are indicative of model misspecification. Still, we can't be confident that this isn't just an inference failure. After all, we used importance sampling with the prior as our proposal distribution. Since the prior over slopes and intercepts is very narrow, it is very likely that *none* of the 2000 inner samples used by [`importance_resampling`](@ref) came close to the true slope and intercept. So it is possible that the issues above arise because importance sampling fails to produce a good approximation of the true posterior.

Before using enumerative inference to resolve this ambiguity, let us try using an MCMC inference algorithm, which might avoid the inference failures of importance sampling by exploring a broader region of the latent space. Similar to the [tutorial on MCMC](@ref mcmc_map_tutorial), we'll use an MCMC kernel that performs Gaussian drift on the continuous parameters, followed by block resimulation on the outlier classifications:

```@example enumerative_tutorial
@gen function line_proposal(trace)
    slope ~ normal(trace[:slope], 0.5)
    intercept ~ normal(trace[:intercept], 0.5)
end

function mcmc_kernel(trace)
    # Gaussian drift on line parameters
    (trace, _) = mh(trace, line_proposal, ())
    
    # Block resimulation: Update the outlier classifications
    (xs,) = get_args(trace)
    n = length(xs)
    for i=1:n
        (trace, _) = mh(trace, select(:data => i => :is_outlier))
    end
    return trace
end

function mcmc_sampler(kernel, trace, n_iters::Int, n_burnin::Int = 0)
    traces = Vector{typeof(trace)}()
    for i in 1:(n_iters + n_burnin)
        trace = kernel(trace)
        if i > n_burnin
            push!(traces, trace)
        end
    end
    return traces
end
nothing # hide
```

In addition, we will intiialize MCMC at the true slope and intercept. This way, we can rule out the possibility that MCMC never explores the region of latent space near the true parameters.

```@example enumerative_tutorial
# Generate initial trace at true slope and intercept
constraints = choicemap(
    :slope => true_slope,
    :intercept => true_intercept
)
constraints = merge(constraints, observations)
(trace, _) = Gen.generate(misspecified_regression, (xs,), constraints)

# Run MCMC for 10,000 iterations with a burn-in of 500
traces = mcmc_sampler(mcmc_kernel, trace, 10000, 500)

# Compute the mean slope and intercept
mean_slope = sum(trace[:slope] for trace in traces) / length(traces)
mean_intercept = sum(trace[:intercept] for trace in traces) / length(traces)

println("Mean slope: ", mean_slope)
println("Mean intercept: ", mean_intercept)
```

Like importance sampling, MCMC infers a much smaller slope and intercept than the true parameters. Let us visualize the joint posterior.

```@example enumerative_tutorial
# Plot posterior samples
slopes = [trace[:slope] for trace in traces]
intercepts = [trace[:intercept] for trace in traces]
plot_posterior_samples(-4:0.25:4, -4:0.25:4, intercepts, slopes,
                       x_true=true_intercept, y_true=true_slope,
                       xlabel="Intercept", ylabel="Slope")
```

Let us also plot the inferred outlier probabilities:

```@example enumerative_tutorial
# Estimate posterior probability of each point being an outlier
outlier_probs = zeros(length(xs))
for i = 1:length(xs)
    for (j, trace) in enumerate(traces)
        if trace[:data => i => :is_outlier]
            outlier_probs[i] += 1/length(traces)
        end
    end
end

# Plot posterior probability of each point being an outlier
bar(1:length(xs), outlier_probs, 
    xlabel="x", ylabel="P(outlier | data)",
    color=:black, ylim=(0, 1), legend=false)
```

Both MCMC and importance sampling produce similar inferences, inferring a flat slope with many outliers rather than a steep slope with few outliers. This is despite the fact that MCMC was initialized at the true parameters, strongly indicating that model misspecification is at play here.

In general, however, we don't have access to the true parameters, nor do we always know if MCMC will converge to the posterior given a finite sample budget. To decisively diagnose model misspecification, we now use enumerative inference with a sufficiently fine grid, ensuring systemic coverage over the latent space.

```@example enumerative_tutorial
# Construct enumeration grid
grid = choice_vol_grid(
    (:slope, -4:0.25:4, :continuous),  # 32 grid intervals
    (:intercept, -4:0.25:4, :continuous),  # 32 grid intervals
    # Enumerate exactly over outlier classifications
    (:data => 1 => :is_outlier, [false, true]),
    (:data => 2 => :is_outlier, [false, true]),
    (:data => 3 => :is_outlier, [false, true]),
    (:data => 4 => :is_outlier, [false, true]),
    (:data => 5 => :is_outlier, [false, true]);
    anchor = :midpoint # Anchor evaluation point at midpoint of each interval
)

# Run enumerative inference
traces, log_norm_weights, lml_est = 
    enumerative_inference(misspecified_regression, (xs,), observations, grid)

# Compute marginal posterior over slope and intercept
sum_dims = Tuple(3:ndims(log_norm_weights)) # Sum over all other variables
posterior_grid = sum(exp.(log_norm_weights), dims=sum_dims)
posterior_grid = dropdims(posterior_grid; dims=sum_dims)

# Extract parameter ranges
slope_range = [trs[1][:slope] for trs in eachslice(traces, dims=1)]
intercept_range = [trs[1][:intercept] for trs in eachslice(traces, dims=2)]

# Plot 2D posterior heatmap with 1D marginals as histograms
plot_posterior_grid(intercept_range, slope_range, posterior_grid,
                    x_true = true_intercept, y_true = true_slope,
                    xlabel = "Intercept", ylabel = "Slope", is_discrete = false)
```

While enumerative inference produces a posterior approximation that is smoother than both importance sampling and MCMC, it still assigns a very low posterior density to the true slope and intercept. Inspecting the outlier classifications, we see that many points are inferred as likely outliers:

```@example enumerative_tutorial
# Compute posterior probability of each point being an outlier
outlier_probs = zeros(length(xs))
for (j, trace) in enumerate(traces)
    for i in 1:length(xs)
        if trace[:data => i => :is_outlier]
            outlier_probs[i] += exp(log_norm_weights[j])
        end
    end
end

bar(1:length(xs), outlier_probs, 
    xlabel="x", ylabel="P(outlier | data)",
    color=:black, ylim=(0, 1), legend=false)
```

This confirms that *model misspecification is the underlying issue*: The generative model we wrote doesn't capture our intuitions about what posterior inference from the data should give us.

## Addressing Model Misspecification

Now that we know our model is misspecified, how do we fix it? In the specific example we considered, the priors over the slope and intercept are too narrow, whereas the outlier probability is too high. A straightforward fix would thus be to widen the slope and intercept priors, while lowering the outlier probability.

However, this change might not generalize to other sets of observations. If some data really is well-characterized by a shallow slope with many outliers, we would like to infer this as well. A more robust solution then, is to introduce *hyper-priors*: Priors on the parameters of the slope and intercept priors and the outlier probability. Adding hyper-priors results in a hierarchical Bayesian model:

```@example enumerative_tutorial
@gen function h_bayes_regression(xs::Vector{<:Real})
    # Hyper-prior on slope and intercept prior variances
    slope_var ~ inv_gamma(1, 1)
    intercept_var ~ inv_gamma(1, 1)
    # Slope and intercept priors
    slope ~ normal(0, sqrt(slope_var))
    intercept ~ normal(0, sqrt(intercept_var))
    # Prior on outlier probability
    prob_outlier ~ beta(1, 1)
    
    # Sample outlier classification and y value for each x value
    n = length(xs)
    ys = Float64[]
    for i = 1:n
        # Sample outlier classification
        is_outlier = {:data => i => :is_outlier} ~ bernoulli(prob_outlier)
        
        if is_outlier
            # Outliers have large noise
            y = {:data => i => :y} ~ normal(0., 5.)
        else
            # Inliers follow the linear relationship, with low noise
            y_mean = slope * xs[i] + intercept
            y = {:data => i => :y} ~ normal(y_mean, 1.)
        end
        push!(ys, y)
    end
    
    return ys
end
nothing # hide
```

Let's run enumerative inference on this expanded model, using a coarser grid to compensate for the increased dimensionality of the latent space:

```@example enumerative_tutorial
# Construct enumeration grid
grid = choice_vol_grid(
    (:slope, -4:1:4, :continuous),  # 8 grid intervals
    (:intercept, -4:1:4, :continuous),  # 8 grid intervals
    (:slope_var, 0:1:5, :continuous),  # 5 grid intervals
    (:intercept_var, 0:1:5, :continuous),  # 5 grid intervals
    (:prob_outlier, 0.0:0.2:1.0, :continuous),  # 5 grid intervals
    # Enumerate exactly over outlier classifications
    (:data => 1 => :is_outlier, [false, true]),
    (:data => 2 => :is_outlier, [false, true]),
    (:data => 3 => :is_outlier, [false, true]),
    (:data => 4 => :is_outlier, [false, true]),
    (:data => 5 => :is_outlier, [false, true]);
    anchor = :midpoint # Anchor evaluation point at midpoint of each interval
)

# Run enumerative inference (this may take a while)
traces, log_norm_weights, lml_est = 
    enumerative_inference(h_bayes_regression, (xs,), observations, grid)

# Compute marginal posterior over slope and intercept
sum_dims = Tuple(3:ndims(log_norm_weights)) # Sum over all other variables
posterior_grid = sum(exp.(log_norm_weights), dims=sum_dims)
posterior_grid = dropdims(posterior_grid; dims=sum_dims)

# Extract parameter ranges
slope_range = [trs[1][:slope] for trs in eachslice(traces, dims=1)]
intercept_range = [trs[1][:intercept] for trs in eachslice(traces, dims=2)]

# Plot 2D posterior heatmap with 1D marginals as histograms
plot_posterior_grid(intercept_range, slope_range, posterior_grid,
                    x_true = true_intercept, y_true = true_slope,
                    xlabel = "Intercept", ylabel = "Slope", is_discrete = false)
```

We see that the mode of the posterior distribution is now close to the true parameters (though there is also a secondary mode corresponding to the interpretation that the data has a shallow slope with outliers). To get a sense of why inference is now reasonable under our new model, let us visualize the conditional posteriors over `slope_var`, `intercept_var` and `prob_outlier` when `slope` and `intercept` are fixed at their true values.

```@example enumerative_tutorial
# Construct enumeration grid conditional on true slope and intercept
cond_grid = choice_vol_grid(
    (:slope_var, 0.0:0.5:5.0, :continuous),  # 10 grid intervals
    (:intercept_var, 0.0:0.5:5.0, :continuous),  # 10 grid intervals
    (:prob_outlier, 0.0:0.1:1.0, :continuous),  # 10 grid intervals
    # Enumerate exactly over outlier classifications
    (:data => 1 => :is_outlier, [false, true]),
    (:data => 2 => :is_outlier, [false, true]),
    (:data => 3 => :is_outlier, [false, true]),
    (:data => 4 => :is_outlier, [false, true]),
    (:data => 5 => :is_outlier, [false, true]);
    anchor = :midpoint # Anchor evaluation point at the right of each interval
)

# Run enumerative inference over conditional posterior
constraints = choicemap(:slope => true_slope, :intercept => true_intercept)
constraints = merge(constraints, observations)
traces, log_norm_weights, lml_est = 
    enumerative_inference(h_bayes_regression, (xs,), constraints, cond_grid)

# Compute marginal posterior over slope_var and intercept_var
sum_dims = Tuple(3:ndims(log_norm_weights)) # Sum over all other variables
posterior_grid = sum(exp.(log_norm_weights), dims=sum_dims)
posterior_grid = dropdims(posterior_grid; dims=sum_dims)

# Extract parameter ranges
slope_var_range = [trs[1][:slope_var] for trs in eachslice(traces, dims=1)]
intercept_var_range = [trs[1][:intercept_var] for trs in eachslice(traces, dims=2)]

# Plot 2D posterior heatmap with 1D marginals as histograms
plot_posterior_grid(intercept_var_range, slope_var_range, posterior_grid,
                    xlabel = "Intercept Variance", ylabel = "Slope Variance",
                    is_discrete = false)
```

```@example enumerative_tutorial
# Compute marginal posterior over prob_outlier
sum_dims = (1, 2, 4:ndims(log_norm_weights)...) # Sum over all other variables
prob_outlier_grid = sum(exp.(log_norm_weights), dims=sum_dims)
prob_outlier_grid = dropdims(prob_outlier_grid; dims=sum_dims)
prob_outlier_range = [trs[1][:prob_outlier] for trs in eachslice(traces, dims=3)]

# Plot marginal posterior distribution over prob_outlier
bar(prob_outlier_range, prob_outlier_grid, 
    legend=false, bar_width=diff(prob_outlier_range)[1],
    linewidth=0, color=:black, widen=false, xlims=(0, 1),
    xlabel = "Outlier Probability (prob_outlier)",
    ylabel = "Conditional Posterior Probability")
```

Conditional on the observed data and the true parameters (`slope = 2.8` and `intercept = -2.4`), the distribution over `slope_var` and `intercept_var` skews towards large values, while the distribution over `prob_outlier` skews towards low values. This avoids the failure mode that arose when the slope and intercept priors were forced to be narrow. Instead, `slope_var`, `intercept_var` and `prob_outlier` can adjust upwards or downwards to adapt to the observed data.

Having gained confidence that our new model is well-specified by performing enumerative inference at a coarse-grained level, we can now use MCMC to approximate the posterior more efficiently, and with a higher degree of spatial resolution.

```@example enumerative_tutorial
function h_bayes_mcmc_kernel(trace)
    # Gaussian drift on line parameters
    (trace, _) = mh(trace, line_proposal, ())

    # Block resimulation: Update the outlier classifications
    (xs,) = get_args(trace)
    n = length(xs)
    for i=1:n
        (trace, _) = mh(trace, select(:data => i => :is_outlier))
    end
    
    # Block resimulation: Update the prior parameters
    (trace, _) = mh(trace, select(:slope_var))
    (trace, _) = mh(trace, select(:intercept_var))
    (trace, _) = mh(trace, select(:prob_outlier))
    return trace
end

# Generate initial trace from prior
trace, _ = Gen.generate(h_bayes_regression, (xs,), observations)

# Run MCMC for 20,000 iterations with a burn-in of 500
traces = mcmc_sampler(h_bayes_mcmc_kernel, trace, 20000, 500)

# Plot posterior samples
slopes = [trace[:slope] for trace in traces]
intercepts = [trace[:intercept] for trace in traces]
plot_posterior_samples(-4:0.25:4, -4:0.25:4, intercepts, slopes,
                       x_true=true_intercept, y_true=true_slope,
                       xlabel="Intercept", ylabel="Slope")
```

MCMC produces samples that concentrate around the true parameters, while still exhibiting some of the bimodality we saw when using coarse-grained enumerative inference.
