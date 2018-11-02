using Gen
using PyPlot
using Printf: @sprintf
import Random

include("scenes.jl")
include("path_planner.jl")

# depends on: scenes.jl, path_planner.jl


#####################################
# lightweight implementation of hmm #
#####################################

@gen function lightweight_hmm(step::Int, path::Path, distances_from_start::Vector{Float64},
                              times::Vector{Float64}, speed::Float64,
                              noise::Float64, dist_slack::Float64)

    # walk path
	locations = Vector{Point}(undef, length(times))
	dist = @addr(normal(speed * times[1], dist_slack), (:dist, 1))
	locations[1] = walk_path(path, distances_from_start, dist)
	for t=2:step
		dist = @addr(normal(dist + speed * (times[t] - times[t-1]), dist_slack), (:dist, t))
		locations[t] = walk_path(path, distances_from_start, dist)
	end

    # generate noisy observations
    for i=1:step
        point = locations[i]
        @addr(normal(point.x, noise), (i, :x))
        @addr(normal(point.y, noise), (i, :y))
    end

    return locations
end


##################################
# compiled implementation of hmm #
##################################

struct KernelState
    dist::Float64
    loc::Point
end

struct KernelParams
    times::Vector{Float64}
    path::Path
    distances_from_start::Vector{Float64}
    speed::Float64
    dist_slack::Float64
    noise::Float64
end

function dist_mean(prev_state::KernelState, params::KernelParams, t::Int)
    if t > 1
        prev_state.dist + (params.times[t] - params.times[t-1]) * params.speed
    else
        params.speed * params.times[1]
    end
end

@compiled @gen function kernel(t::Int, prev_state::KernelState, params::KernelParams)
    # NOTE: if t = 1 then prev_state will have all NaNs
    dist::Float64 = @addr(normal(dist_mean(prev_state, params, t), params.dist_slack), :dist)
    loc::Point = walk_path(params.path, params.distances_from_start, dist)
    @addr(normal(loc.x, params.noise), :x)
    @addr(normal(loc.y, params.noise), :y)
    return KernelState(dist, loc)::KernelState
end

# construct a hidden markov model generator
# args: (num_steps::Int, init_state::KernelState, params::KernelParams)
compiled_hmm = markov(kernel)

# NOTE: you have to separate compiled gen function definitions from calling API
# methods on them with a top-level call to this function
Gen.load_generated_functions()


#############################
# define some fixed context #
#############################

function make_scene()
    scene = Scene(0, 1, 0, 1) 
    add!(scene, Tree(Point(0.30, 0.20), size=0.1))
    add!(scene, Tree(Point(0.83, 0.80), size=0.1))
    add!(scene, Tree(Point(0.80, 0.40), size=0.1))
    horiz = 1
    vert = 2
    wall_height = 0.30
    wall_thickness = 0.02
    walls = [
        Wall(Point(0.20, 0.40), horiz, 0.40, wall_thickness, wall_height)
        Wall(Point(0.60, 0.40), vert, 0.40, wall_thickness, wall_height)
        Wall(Point(0.60 - 0.15, 0.80), horiz, 0.15 + wall_thickness, wall_thickness, wall_height)
        Wall(Point(0.20, 0.80), horiz, 0.15, wall_thickness, wall_height)
        Wall(Point(0.20, 0.40), vert, 0.40, wall_thickness, wall_height)]
    for wall in walls
        add!(scene, wall)
    end
    return scene
end

const scene = make_scene()
const times = collect(range(0, stop=1, length=20))
const start_x = 0.1
const start_y = 0.1
const stop_x = 0.5
const stop_y = 0.5
const speed = 0.5
const noise = 0.02
const dist_slack = 0.2
const start = Point(start_x, start_y)
const stop = Point(stop_x, stop_y)


####################
# trace renderings #
####################

# the only difference between rendering the lightweight and compiled versions
# how the locations are extracted from the trace, and the addresses of the
# measurements.

function render_lightweight_hmm_trace(scene::Scene, start::Point, stop::Point,
                maybe_path::Nullable{Path},
                times::Vector{Float64}, speed::Float64,
                noise::Float64, dist_slack::Float64, trace, ax;
                show_measurements=true,
                show_start=true, show_stop=true,
                show_path=true, show_noise=true,
                start_alpha=1., stop_alpha=1., path_alpha=1.)

    # set current axis
    sca(ax)

    # render obstacles
    render(scene, ax)

    # plot start and stop
    if show_start
        scatter([start.x], [start.y], color="blue", s=100, alpha=start_alpha)
    end
    if show_stop
        scatter([stop.x], [stop.y], color="red", s=100, alpha=stop_alpha)
    end

    # plot path lines
    if !isnull(maybe_path) && show_path
        path = get(maybe_path)
        for i=1:length(path.points)-1
            prev = path.points[i]
            next = path.points[i+1]
            plot([prev.x, next.x], [prev.y, next.y], color="black", alpha=0.5, linewidth=5)
        end
    end

    # plot locations with measurement noise around them
    locations = get_call_record(trace).retval
    scatter([pt.x for pt in locations], [pt.y for pt in locations],
        color="orange", alpha=path_alpha, s=25)
    if show_noise
        for pt in locations
            circle = patches[:Circle]((pt.x, pt.y), noise, facecolor="purple", alpha=0.2)
            ax[:add_patch](circle)
        end
    end
    
    # plot measured locations
    assignment = get_assignment(trace)
    if show_measurements
        measured_xs = [assignment[(i, :x)] for i=1:length(locations)]
        measured_ys = [assignment[(i, :y)] for i=1:length(locations)]
        scatter(measured_xs, measured_ys, marker="x", color="black", alpha=1., s=25)
    end
end

function render_compiled_hmm_trace(scene::Scene, start::Point, stop::Point,
                maybe_path::Nullable{Path},
                times::Vector{Float64}, speed::Float64,
                noise::Float64, dist_slack::Float64, trace, ax;
                show_measurements=true,
                show_start=true, show_stop=true,
                show_path=true, show_noise=true,
                start_alpha=1., stop_alpha=1., path_alpha=1.)

    # set current axis
    sca(ax)

    # render obstacles
    render(scene, ax)

    # plot start and stop
    if show_start
        scatter([start.x], [start.y], color="blue", s=100, alpha=start_alpha)
    end
    if show_stop
        scatter([stop.x], [stop.y], color="red", s=100, alpha=stop_alpha)
    end

    # plot path lines
    if !isnull(maybe_path) && show_path
        path = get(maybe_path)
        for i=1:length(path.points)-1
            prev = path.points[i]
            next = path.points[i+1]
            plot([prev.x, next.x], [prev.y, next.y], color="black", alpha=0.5, linewidth=5)
        end
    end

    # plot locations with measurement noise around them
    states = get_call_record(trace).retval
    locations = Point[state.loc for state in states]
    scatter([pt.x for pt in locations], [pt.y for pt in locations],
        color="orange", alpha=path_alpha, s=25)
    if show_noise
        for pt in locations
            circle = patches[:Circle]((pt.x, pt.y), noise, facecolor="purple", alpha=0.2)
            ax[:add_patch](circle)
        end
    end
    
    # plot measured locations
    assignment = get_assignment(trace)
    if show_measurements
        measured_xs = [assignment[i => :x] for i=1:length(locations)]
        measured_ys = [assignment[i => :y] for i=1:length(locations)]
        scatter(measured_xs, measured_ys, marker="x", color="black", alpha=1., s=25)
    end
end


######################
# show prior samples #
######################

function show_prior_samples()
    Random.seed!(0)

    # generate a path
    maybe_path = plan_path(start, stop, scene, PlannerParams(300, 3.0, 2000, 1.))
    @assert !isnull(maybe_path)
    path = get(maybe_path)
 	distances_from_start = compute_distances_from_start(path)

    # show samples from the lightweight model
    model_args = (length(times), path, distances_from_start, times, speed, noise, dist_slack)
    figure(figsize=(32, 32))
    for i=1:15
        subplot(4, 4, i)
        ax = gca()
        trace = simulate(lightweight_hmm, model_args)
        render_lightweight_hmm_trace(scene, start, stop, maybe_path, times, speed, noise, dist_slack, trace, ax)
    end
    savefig("lightweight_hmm_prior_samples.png")

    # show samples from the compiled model
    kernel_params = KernelParams(times, path, distances_from_start, speed, dist_slack, noise)
    model_args_rest = (KernelState(NaN, Point(NaN, NaN)), kernel_params)
    figure(figsize=(32, 32))
    for i=1:15
        subplot(4, 4, i)
        ax = gca()
        trace = simulate(compiled_hmm, (length(times), model_args_rest...))
        render_compiled_hmm_trace(scene, start, stop, maybe_path, times, speed, noise, dist_slack, trace, ax)
    end
    savefig("compiled_hmm_prior_samples.png")

end
show_prior_samples()


##################################
# particle filtering experiments #
##################################

function experiment()

    Random.seed!(0)

    # generate a path and precompute distances from the start
    maybe_path = plan_path(start, stop, scene, PlannerParams(300, 3.0, 2000, 1.))
    @assert !isnull(maybe_path)
    path = get(maybe_path)
 	distances_from_start = compute_distances_from_start(path)

    # generate ground truth locations and observations
    model_args = (length(times), path, distances_from_start, times, speed, noise, dist_slack)
    trace = simulate(lightweight_hmm, model_args)
    assignment = get_assignment(trace)
    measured_xs = [assignment[(i, :x)] for i=1:length(times)]
    measured_ys = [assignment[(i, :y)] for i=1:length(times)]

    # all particles
    num_particles_list = [10, 30, 100, 300, 1000]
 
    ## particle filtering in lightweight hmm
    println("lightweight hmm..")

    function get_lightweight_hmm_obs(step::Int)
        assignment = DynamicAssignment()
        assignment[(step, :x)] = measured_xs[step]
        assignment[(step, :y)] = measured_ys[step]
        return assignment
    end

    num_reps = 20
    model_args_rest = (path, distances_from_start, times, speed, noise, dist_slack)
    results_lightweight = Dict{Int, Tuple{Vector{Float64},Vector{Float64}}}()
    for num_particles in num_particles_list
        ess_threshold = num_particles / 2
        elapsed = Vector{Float64}(undef, num_reps)
        lmls = Vector{Float64}(undef, num_reps)
        for rep=1:num_reps
            start = time_ns()
            (traces, _, lml) = particle_filter(lightweight_hmm, model_args_rest, length(times),
                                                     num_particles, ess_threshold,
                                                     get_lightweight_hmm_obs)
            elapsed[rep] = Int(time_ns() - start) / 1000000000
            println("num_particles: $num_particles, lml estimate: $lml")
            lmls[rep] = lml
        end
        results_lightweight[num_particles] = (lmls, elapsed)
    end

    ## particle filtering in compilled hmm
    println("compiled hmm..")

    function get_compiled_hmm_obs(step::Int)
        assignment = DynamicAssignment()
        assignment[step => :x] = measured_xs[step]
        assignment[step => :y] = measured_ys[step]
        return assignment
    end

    num_reps = 20
    kernel_params = KernelParams(times, path, distances_from_start, speed, dist_slack, noise)
    model_args_rest = (KernelState(NaN, Point(NaN, NaN)), kernel_params)
    results_compiled = Dict{Int, Tuple{Vector{Float64},Vector{Float64}}}()
    for num_particles in num_particles_list
        ess_threshold = num_particles / 2
        elapsed = Vector{Float64}(undef, num_reps)
        lmls = Vector{Float64}(undef, num_reps)
        for rep=1:num_reps
            start = time_ns()
            (traces, _, lml) = particle_filter(compiled_hmm, model_args_rest, length(times),
                                                     num_particles, ess_threshold,
                                                     get_compiled_hmm_obs)
            elapsed[rep] = Int(time_ns() - start) / 1000000000
            println("num_particles: $num_particles, lml estimate: $lml")
            lmls[rep] = lml
        end
        results_compiled[num_particles] = (lmls, elapsed)
    end

    # plot results

    figure(figsize=(8, 8))

    median_elapsed = [median(results_lightweight[num_particles][2]) for num_particles in num_particles_list]
    mean_lmls = [mean(results_lightweight[num_particles][1]) for num_particles in num_particles_list]
    plot(median_elapsed, mean_lmls, label="lightweight", color="blue")

    median_elapsed = [median(results_compiled[num_particles][2]) for num_particles in num_particles_list]
    mean_lmls = [mean(results_compiled[num_particles][1]) for num_particles in num_particles_list]
    plot(median_elapsed, mean_lmls, label="compiled", color="orange")

    ax = gca()
    ax[:set_xscale]("log")
    xlabel("runtime (sec.)")
    ylabel("log marginal likelihood est.")
    savefig("filtering_results.png")
end

experiment()
