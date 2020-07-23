---
layout: splash
---
<br>

# Tutorial: Particle Filtering in Gen

Sequential Monte Carlo (SMC) methods such as particle filtering iteratively solve a *sequence of inference problems* using techniques based on importance sampling and in some cases MCMC. The solution to each problem in the sequence is represented as a collection of samples or *particles*. The particles for each problem are based on extending or adjusting the particles for the previous problem in the sequence.

The sequence of inference problems that are solved often arise naturally from observations that arrive incrementally, as in particle filtering. The problems can also be constructed instrumentally to facilitate inference, as in annealed importance sampling [3]. This tutorial shows how to use Gen to implement a particle filter for a tracking problem that uses "rejuvenation" MCMC moves. Specifically, we will address the "bearings only tracking" problem described in [4].

[1] Doucet, Arnaud, Nando De Freitas, and Neil Gordon. "An introduction to sequential Monte Carlo methods." Sequential Monte Carlo methods in practice. Springer, New York, NY, 2001. 3-14.

[2] Del Moral, Pierre, Arnaud Doucet, and Ajay Jasra. "Sequential monte carlo samplers." Journal of the Royal Statistical Society: Series B (Statistical Methodology) 68.3 (2006): 411-436.

[3] Neal, Radford M. "Annealed importance sampling." Statistics and computing 11.2 (2001): 125-139.

[4] Gilks, Walter R., and Carlo Berzuini. "Following a moving target—Monte Carlo inference for dynamic Bayesian models." Journal of the Royal Statistical Society: Series B (Statistical Methodology) 63.1 (2001): 127-146. [PDF](http://www.mathcs.emory.edu/~whalen/Papers/BNs/MonteCarlo-DBNs.pdf)

## Outline

**Section 1**: [Implementing the generative model](#basic-model)

**Section 2**: [Implementing a basic particle filter](#basic-pf)

**Section 3**: [Adding rejuvenation moves](#rejuv)

**Section 4**: [Using the unfold combinator to improve performance](#unfold)


```julia
using Gen
using PyPlot
```

## 1. Implementing the generative model <a name="basic-model"></a>

We will implement a generative model for the movement of a point in the x-y plane and bearing measurements of the location of this point relative to the origin over time.

We assume that we know the approximate initial position and velocity of the point. We assume the point's x- and y- velocity are subject to random perturbations drawn from some normal distribution with a known variance. Each bearing measurement consists of the angle of the point being tracked relative to the positive x-axis.

We write the generative model as a generative function below. The function first samples the initial state of the point from a prior distribution, and then generates `T` successive states in a `for` loop. The argument to the model is the number of states not including the initial state.


```julia
bearing(x, y) = atan(y, x)

@gen function model(T::Int)
    
    measurement_noise = 0.005
    velocity_var = (1.0/1e6)

    xs = Vector{Float64}(undef, T+1)
    ys = Vector{Float64}(undef, T+1)

    # prior on initial x-coordinate
    x = @trace(normal(0.01, 0.01), :x0)
       
    # prior on initial y-coordinate
    y = @trace(normal(0.95, 0.01), :y0)
    
    # prior on x-component of initial velocity
    vx = @trace(normal(0.002, 0.01), :vx0)
    
    # prior on y-component of initial velocity
    vy = @trace(normal(-0.013, 0.01), :vy0)
    
    # initial bearing measurement
    @trace(normal(bearing(x, y), measurement_noise), :z0)

    # record position
    xs[1] = x
    ys[1] = y
    
    # generate successive states and measurements
    for t=1:T
        
        # update the state of the point
        vx = @trace(normal(vx, sqrt(velocity_var)), (:vx, t))
        vy = @trace(normal(vy, sqrt(velocity_var)), (:vy, t))
        x += vx
        y += vy
        
        # bearing measurement
        @trace(normal(bearing(x, y), measurement_noise), (:z, t))

        # record position
        xs[t+1] = x
        ys[t+1] = y
    end
    
    # return the sequence of positions
    return (xs, ys)
end;
```

We generate a data set of positions, and observed bearings, by sampling from this model, with `T=50`:


```julia
import Random
Random.seed!(4)

# generate trace with specific initial conditions
T = 50
constraints = Gen.choicemap((:x0, 0.01), (:y0, 0.95), (:vx0, 0.002), (:vy0, -0.013))
(trace, _) = Gen.generate(model, (T,), constraints)

# extract the observed data (zs) from the trace
choices = Gen.get_choices(trace)
zs = Vector{Float64}(undef, T+1)
zs[1] = choices[:z0]
for t=1:T
    zs[t+1] = choices[(:z, t)]
end
```

We next write a visualization for traces of this model below. It shows the positions and dots and the observed bearings as lines from the origin:


```julia
function render(trace; show_data=true, max_T=get_args(trace)[1])
    (T,) = Gen.get_args(trace)
    choices = Gen.get_choices(trace)
    (xs, ys) = Gen.get_retval(trace)
    zs = Vector{Float64}(undef, T+1)
    zs[1] = choices[:z0]
    for t=1:T
        zs[t+1] = choices[(:z, t)]
    end
    scatter(xs[1:max_T+1], ys[1:max_T+1], s=5)
    if show_data
        for z in zs[1:max_T+1]
            dx = cos(z) * 0.5
            dy = sin(z) * 0.5
            plot([0., dx], [0., dy], color="red", alpha=0.3)
        end
    end
end;
```

We visualize the synthetic trace below:


```julia
render(trace)
```


![png](output_9_0.png)


## 2. Implementing a basic particle filter <a name="basic-pf"></a>

In Gen, a **particle is represented as a trace** and the particle filter state contains a weighted collection of traces. Below we define an inference program that runs a particle filter on an observed data set of bearings (`zs`). We use `num_particles` particles internally, and then we return a sample of `num_samples` traces from the weighted collection that the particle filter produces.

Gen provides methods for initializing and updating the state of a particle filter, documented in [Particle Filtering](https://probcomp.github.io/Gen/dev/ref/inference/#Particle-Filtering-1).

- `Gen.initialize_particle_filter`

- `Gen.particle_filter_step!`

Both of these methods can used either with the default proposal or a custom proposal. In this tutorial, we will use the default proposal. There is also a method that resamples particles based on their weights, which serves to redistribute the particles to more promising parts of the latent space.

- `Gen.maybe_resample!`

Gen also provides a method for sampling a collection of unweighted traces from the current weighted collection in the particle filter state:

- `Gen.sample_unweighted_traces`


```julia
function particle_filter(num_particles::Int, zs::Vector{Float64}, num_samples::Int)
    
    # construct initial observations
    init_obs = Gen.choicemap((:z0, zs[1]))
    state = Gen.initialize_particle_filter(model, (0,), init_obs, num_particles)
    
    # steps
    for t=1:length(zs)-1
        Gen.maybe_resample!(state, ess_threshold=num_particles/2)
        obs = Gen.choicemap(((:z, t), zs[t+1]))
        Gen.particle_filter_step!(state, (t,), (UnknownChange(),), obs)
    end
    
    # return a sample of unweighted traces from the weighted collection
    return Gen.sample_unweighted_traces(state, num_samples)
end;
```

The initial state is obtained by providing the following to `initialize_particle_filter`:

- The generative function for the generative model (`model`)

- The initial arguments to the generative function.

- The initial observations, expressed as a map from choice address to values (`init_obs`).

- The number of particles.

At each step, we resample from the collection of traces (`maybe_resample!`) and then we introduce one additional bearing measurement by calling `particle_filter_step!` on the state. We pass the following arguments to `particle_filter_step!`:

- The state (it will be mutated)

- The new arguments to the generative function for this step. In our case, this is the number of measurements beyond the first measurement.

- The [argdiff](https://probcomp.github.io/Gen/dev/ref/gfi/#Argdiffs-1) value, which provides detailed information about the change to the arguments between the previous step and this step. We will revisit this value later. For now, we indicat ethat we do not know how the `T::Int` argument will change with each step.

- The new observations associated with the new step. In our case, this just contains the latest measurement.


We run this particle filter with 5000 particles, and return a sample of 100 particles. This will take 30-60 seconds. We will see one way of speeding up the particle filter in a later section.


```julia
@time pf_traces = particle_filter(5000, zs, 200);
```

     36.481359 seconds (134.90 M allocations: 5.846 GiB, 47.35% gc time)


To render these traces, we first define a function that overlays many renderings:


```julia
function overlay(renderer, traces; same_data=true, args...)
    renderer(traces[1], show_data=true, args...)
    for i=2:length(traces)
        renderer(traces[i], show_data=!same_data, args...)
    end
end;
```

We then render the traces from the particle filter:


```julia
overlay(render, pf_traces, same_data=true)
```


![png](output_18_0.png)


Notice that as during the period of denser bearing measurements, the trajectories tend to turn so that the heading is more parallel to the bearing vector. An alternative explanation is that the point maintained a constant heading, but just slowed down significantly. It is interesting to see that the inferences favor the "turning explanation" over the "slowing down explanation".

----

### Exercise
Run the particle filter with fewer particles and visualize the results.

### Solution

------

## 3. Adding rejuvenation moves <a name="rejuv"></a>

It is sometimes useful to add MCMC moves to particles in a particle filter between steps. These MCMC moves are often called "rejuvenation moves" [4]. Each rejuvenation moves targets the *current posterior distribution* at the given step. For example, when applying the rejuvenation move after incorporating 3 observations, our rejuvenation moves have as their stationary distribution the conditional distribution on the latent variables, given the first three observations.

Next, we write a version of the particle filter that applies two random walk Metropolis-Hastings rejuvenation move to each particle.

The cell below defines a Metropolis-Hastings perturbation move that perturbs the velocity vectors for a block of time steps between `a` and `b` inclusive.


```julia
@gen function perturbation_proposal(prev_trace, a::Int, b::Int)
    choices = get_choices(prev_trace)
    (T,) = get_args(prev_trace)
    for t=a:b
        vx = @trace(normal(choices[(:vx, t)], 1e-3), (:vx, t))
        vy = @trace(normal(choices[(:vy, t)], 1e-3), (:vy, t))
    end
end

function perturbation_move(trace, a::Int, b::Int)
    Gen.metropolis_hastings(trace, perturbation_proposal, (a, b))
end;
```

We add this into our particle filtering inference program below. We apply the rejuvenation move to adjust the velocities for the previous 5 time steps.


```julia
function particle_filter_rejuv(num_particles::Int, zs::Vector{Float64}, num_samples::Int)
    init_obs = Gen.choicemap((:z0, zs[1]))    
    state = Gen.initialize_particle_filter(model, (0,), init_obs, num_particles)
    for t=1:length(zs)-1
        
        # apply a rejuvenation move to each particle
        for i=1:num_particles
            state.traces[i], _ = perturbation_move(state.traces[i], max(1, t-5), t-1)
        end
        
        Gen.maybe_resample!(state, ess_threshold=num_particles/2)
        obs = Gen.choicemap(((:z, t), zs[t+1]))
        Gen.particle_filter_step!(state, (t,), (UnknownChange(),), obs)
    end
    
    # return a sample of unweighted traces from the weighted collection
    return Gen.sample_unweighted_traces(state, num_samples)
end;
```

We run the particle filter with rejuvenation below. This will take a minute or two. We will see one way of speeding up the particle filter in a later section.


```julia
@time pf_rejuv_traces = particle_filter_rejuv(5000, zs, 200);
```

     72.251390 seconds (334.61 M allocations: 14.156 GiB, 34.14% gc time)


We render the traces:


```julia
overlay(render, pf_rejuv_traces, same_data=true)
```


![png](output_31_0.png)


## 4. Using the unfold combinator to improve performance <a name="unfold"></a>

For the particle filtering algorithms above, within an update step it is only necessary to revisit the most recent state (or the most recent 5 states if the rejuvenation moves are used) because the initial states are never updated, and the contribution of these states to the weight computation cancel.

However, each update step of the particle filter inference programs above scales *linearly* in the size of the trace because it visits every state when computing the weight update. This is because the built-in modeling DSL by default always performs an end-to-end execution of the generative function body whenever performing a trace update. This allows the built-in modeling DSL to be very flexible and to have a simple implementation, at the cost of performance. There are several ways of improving performance after one has a prototype written in the built-in modeling DSL. One of these is [Generative Function Combinators](https://probcomp.github.io/Gen/dev/ref/combinators/), which make the flow of information through the generative process more explicit to Gen, and enable asymptotically more efficient inference programs.

To exploit the opportunity for incremental computation, and improve the scaling behavior of our particle filter inference programs, we will write a new model that replaces the following Julia `for` loop in our model, using a generative function combinator.

```julia
    # generate successive states and measurements
    for t=1:T
        
        # update the state of the point
        vx = @trace(normal(vx, sqrt(velocity_var)), (:vx, t))
        vy = @trace(normal(vy, sqrt(velocity_var)), (:vy, t))
        x += vx
        y += vy
        
        # bearing measurement
        @trace(normal(bearing(x, y), measurement_noise), (:z, t))

        # record position
        xs[t+1] = x
        ys[t+1] = y
    end
```

This `for` loop has a very specific pattern of information flow---there is a sequence of states (represented by `x, y, vx, and vy), and each state is generated from the previous state. This is exactly the pattern that the [Unfold](https://probcomp.github.io/Gen/dev/ref/combinators/#Unfold-combinator-1) generative function combinator is designed to handle.

Below, we re-express the Julia for loop over the state sequence using the Unfold combinator. Specifically, we define a generative function (kernel) that takes the prevous state as its second argument, and returns the new state. The Unfold combinator takes the kernel and returns a new generative function (chain) that applies kernel repeatedly. Read the Unfold combinator documentation for details on the behavior of the resulting generative function (chain).


```julia
struct State
    x::Float64
    y::Float64
    vx::Float64
    vy::Float64
end

@gen (static) function kernel(t::Int, prev_state::State,
                              velocity_var::Float64, measurement_noise::Float64)
    vx = @trace(normal(prev_state.vx, sqrt(velocity_var)), :vx)
    vy = @trace(normal(prev_state.vy, sqrt(velocity_var)), :vy)
    x = prev_state.x + vx
    y = prev_state.y + vy
    @trace(normal(bearing(x, y), measurement_noise), :z)
    next_state = State(x, y, vx, vy)
    return next_state
end

chain = Gen.Unfold(kernel)

Gen.load_generated_functions()
```

We can understand the behavior of `chain` by getting a trace of it and printing the random choices:


```julia
trace = Gen.simulate(chain, (4, State(0., 0., 0., 0.), 0.01, 0.01))
println(Gen.get_choices(trace))
```

    │
    ├── 1
    │   │
    │   ├── :vx : -0.08970902180131173
    │   │
    │   ├── :vy : -0.10682294942385842
    │   │
    │   └── :z : -2.2912702846132453
    │
    ├── 2
    │   │
    │   ├── :vx : -0.1941244343164683
    │   │
    │   ├── :vy : -0.18185481061346262
    │   │
    │   └── :z : -2.340595555861278
    │
    ├── 3
    │   │
    │   ├── :vx : -0.24238833226466439
    │   │
    │   ├── :vy : -0.07267414148918251
    │   │
    │   └── :z : -2.5432083216947494
    │
    └── 4
        │
        ├── :vx : -0.3639453051064175
        │
        ├── :vy : -0.1455945832678338
        │
        └── :z : -2.6233328025606513
    


We now write a new version of the generative model that invokes `chain` instead of using the Julia `for` loop:


```julia
@gen (static) function unfold_model(T::Int)
    
    # parameters
    measurement_noise = 0.005
    velocity_var = 1e-6

    # initial conditions
    x = @trace(normal(0.01, 0.01), :x0)
    y = @trace(normal(0.95, 0.01), :y0)
    vx = @trace(normal(0.002, 0.01), :vx0)
    vy = @trace(normal(-0.013, 0.01), :vy0)

    # initial measurement
    z = @trace(normal(bearing(x, y), measurement_noise), :z0)

    # record initial state
    init_state = State(x, y, vx, vy)
    
    # run `chain` function under address namespace `:chain`, producing a vector of states
    states = @trace(chain(T, init_state, velocity_var, measurement_noise), :chain)
    
    result = (init_state, states)
    return result
end;

Gen.load_generated_functions()
```

Let's generate a trace of this new model program to understand its structure:


```julia
(trace, _) = Gen.generate(unfold_model, (4,))
println(Gen.get_choices(trace))
```

    │
    ├── :x0 : 0.01046259889720223
    │
    ├── :y0 : 0.9480796760211051
    │
    ├── :vx0 : -0.016146187712751288
    │
    ├── :vy0 : -0.021877921008695483
    │
    ├── :z0 : 1.5658875595556638
    │
    └── :chain
        │
        ├── 1
        │   │
        │   ├── :vx : -0.015994703197911624
        │   │
        │   ├── :vy : -0.02199988311438922
        │   │
        │   └── :z : 1.5758174767535693
        │
        ├── 2
        │   │
        │   ├── :vx : -0.017592576483682166
        │   │
        │   ├── :vy : -0.021833363037200983
        │   │
        │   └── :z : 1.601880885130888
        │
        ├── 3
        │   │
        │   ├── :vx : -0.017105404183538057
        │   │
        │   ├── :vy : -0.02134346447535751
        │   │
        │   └── :z : 1.6156207105482778
        │
        └── 4
            │
            ├── :vx : -0.02033475521260161
            │
            ├── :vy : -0.02030158417422561
            │
            └── :z : 1.640019788042328
    



```julia
function unfold_particle_filter(num_particles::Int, zs::Vector{Float64}, num_samples::Int)
    init_obs = Gen.choicemap((:z0, zs[1]))    
    state = Gen.initialize_particle_filter(unfold_model, (0,), init_obs, num_particles)    
    for t=1:length(zs)-1

        maybe_resample!(state, ess_threshold=num_particles/2)
        obs = Gen.choicemap((:chain => t => :z, zs[t+1]))
        Gen.particle_filter_step!(state, (t,), (UnknownChange(),), obs)
    end
    
    # return a sample of traces from the weighted collection:
    return Gen.sample_unweighted_traces(state, num_samples)
end;
```


```julia
@time unfold_pf_traces = unfold_particle_filter(5000, zs, 200);
```

      7.869715 seconds (29.99 M allocations: 1.622 GiB, 20.89% gc time)



```julia
function unfold_render(trace; show_data=true, max_T=get_args(trace)[1])
    (T,) = Gen.get_args(trace)
    choices = Gen.get_choices(trace)
    (init_state, states) = Gen.get_retval(trace)
    xs = Vector{Float64}(undef, T+1)
    ys = Vector{Float64}(undef, T+1)
    zs = Vector{Float64}(undef, T+1)
    xs[1] = init_state.x
    ys[1] = init_state.y
    zs[1] = choices[:z0]
    for t=1:T
        xs[t+1] = states[t].x
        ys[t+1] = states[t].y
        zs[t+1] = choices[:chain => t => :z]
    end
    scatter(xs[1:max_T+1], ys[1:max_T+1], s=5)
    if show_data
        for z in zs[1:max_T+1]
            dx = cos(z) * 0.5
            dy = sin(z) * 0.5
            plot([0., dx], [0., dy], color="red", alpha=0.3)
        end
    end
end;
```

Let's check that the results are reasonable:


```julia
overlay(unfold_render, unfold_pf_traces, same_data=true)
```


![png](output_44_0.png)


We now empirically investigate the scaling behavior of (1) the inference program that uses the Julia `for` loop,  and (2) the equivalent inference program that uses Unfold. We will use a fake long vector of z data, and we will investigate how the running time depends on the number of observations.


```julia
fake_zs = rand(1000);

function timing_experiment(num_observations_list::Vector{Int}, num_particles::Int, num_samples::Int)
    times = Vector{Float64}()
    times_unfold = Vector{Float64}()
    for num_observations in num_observations_list
        println("evaluating inference programs for num_observations: $num_observations")
        tstart = time_ns()
        traces = particle_filter(num_particles, fake_zs[1:num_observations], num_samples)
        push!(times, (time_ns() - tstart) / 1e9)
        
        tstart = time_ns()
        traces = unfold_particle_filter(num_particles, fake_zs[1:num_observations], num_samples)
        push!(times_unfold, (time_ns() - tstart) / 1e9)
        
    end
    (times, times_unfold)
end;

num_observations_list = [1, 3, 10, 30, 50, 100, 150, 200, 500]
(times, times_unfold) = timing_experiment(num_observations_list, 100, 20);
```

    evaluating inference programs for num_observations: 1
    evaluating inference programs for num_observations: 3
    evaluating inference programs for num_observations: 10
    evaluating inference programs for num_observations: 30
    evaluating inference programs for num_observations: 50
    evaluating inference programs for num_observations: 100
    evaluating inference programs for num_observations: 150
    evaluating inference programs for num_observations: 200
    evaluating inference programs for num_observations: 500


Notice that the running time of the inference program without unfold appears to be quadratic in the number of observations, whereas the inference program that uses unfold appears to scale linearly:


```julia
plot(num_observations_list, times, color="blue")
plot(num_observations_list, times_unfold, color="red")
xlabel("Number of observations")
ylabel("Running time (sec.)");
```


![png](output_48_0.png)

