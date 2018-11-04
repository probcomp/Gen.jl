using Gen
using PyPlot
using Printf: @sprintf
import Random
using Statistics: median, mean

include("scenes.jl")
include("path_planner.jl")

# depends on: scenes.jl, path_planner.jl


#####################################
# lightweight implementation of hmm #
#####################################

@gen function lightweight_hmm(step::Int, path::Path, distances_from_start::Vector{Float64},
                              times::Vector{Float64}, speed::Float64,
                              noise::Float64, dist_slack::Float64)
    @assert step >= 1

    # walk path
	locations = Vector{Point}(undef, step)
	dist = @addr(normal(speed * times[1], dist_slack), (:dist, 1))
	locations[1] = walk_path(path, distances_from_start, dist)
	for t=2:step
		dist = @addr(normal(dist + speed * (times[t] - times[t-1]), dist_slack), (:dist, t))
		locations[t] = walk_path(path, distances_from_start, dist)
	end

    # generate noisy observations
    for t=1:step
        point = locations[t]
        @addr(normal(point.x, noise), (:x, t))
        @addr(normal(point.y, noise), (:y, t))
    end

    return locations
end


#################################
# proposals for lightweight hmm #
#################################

@gen function lightweight_init_proposal(speed::Float64, times::Vector{Float64}, dist_slack::Float64)
    @addr(normal(speed * times[1], dist_slack), (:dist, 1))
end

@gen function lightweight_step_proposal(step::Int, prev_dist::Float64,
                                        speed::Float64, times::Vector{Float64}, dist_slack::Float64)
    @assert step > 1
    @addr(normal(prev_dist + speed * (times[step] - times[step-1]), dist_slack), (:dist, step))
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


########
# test #
########

function test()

    Random.seed!(0)

	path = Path(Point(0.1, 0.1), Point(0.5, 0.5), Point[Point(0.1, 0.1), Point(0.0773627, 0.146073), Point(0.167036, 0.655448), Point(0.168662, 0.649074), Point(0.156116, 0.752046), Point(0.104823, 0.838075), Point(0.196407, 0.873581), Point(0.390309, 0.988468), Point(0.408272, 0.91336), Point(0.5, 0.5)])
 	distances_from_start = compute_distances_from_start(path)
	measured_xs = [0.0890091, 0.105928, 0.158508, 0.11927, 0.0674466, 0.0920541, 0.0866197, 0.0828504, 0.0718467, 0.13625, 0.0949714, 0.0903477, 0.0994438, 0.073613, 0.0795392, 0.0993675, 0.111972, 0.0725639, 0.146364, 0.115776]
	measured_ys = [0.25469, 0.506397, 0.452324, 0.377185, 0.23247, 0.110536, 0.11206, 0.115172, 0.170976, 0.0726151, 0.0763815, 0.0888771, 0.0683795, 0.0929964, 0.275081, 0.383367, 0.335842, 0.181095, 0.478705, 0.664434]

    # generate initial trace of 2 steps
    constraints = DynamicAssignment()
    for step=1:2
        constraints[(:x, step)] = measured_xs[step]
        constraints[(:y, step)] = measured_ys[step]
    end
    args = (2, path, distances_from_start, times, speed, noise, dist_slack)
    (trace, _) = generate(lightweight_hmm, args, constraints)
    initial_model_score = get_call_record(trace).score

    # extend to 3 steps
    new_args = (3, path, distances_from_start, times, speed, noise, dist_slack)
    constraints = DynamicAssignment()
    constraints[(:x, 3)] = measured_xs[3]
    constraints[(:y, 3)] = measured_ys[3]
    (new_trace, weight) = extend(lightweight_hmm, new_args, unknownargdiff, trace, constraints)

    # check that we get the same weight
    prev_dist = get_assignment(new_trace)[(:dist, 2)]
    args = (3, prev_dist, speed, times, dist_slack)
    proposal_constraints = DynamicAssignment()
    proposal_constraints[(:dist, 3)] = get_assignment(new_trace)[(:dist, 3)]
    proposal_trace = assess(lightweight_step_proposal, args, proposal_constraints)
    proposal_score = get_call_record(proposal_trace).score
    new_model_score = get_call_record(new_trace).score
    
    println(weight)
    println(new_model_score - initial_model_score - proposal_score)
    @assert isapprox(weight, new_model_score - initial_model_score - proposal_score)
end
test()


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
        measured_xs = [assignment[(:x, i)] for i=1:length(locations)]
        measured_ys = [assignment[(:y, i)] for i=1:length(locations)]
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
    measured_xs = [assignment[(:x, i)] for i=1:length(times)]
    measured_ys = [assignment[(:y, i)] for i=1:length(times)]

    num_particles_list = [10, 30, 100, 300, 1000]
    num_reps = 20
    max_steps = length(times)
    verbose = false

    ## particle filtering in lightweight hmm using internal proposal ##

    println("lightweight hmm..")

    function get_lightweight_hmm_obs(step::Int)
        observations = DynamicAssignment()
        observations[(:x, step)] = measured_xs[step]
        observations[(:y, step)] = measured_ys[step]
        return observations 
    end

    model_args_rest = (path, distances_from_start, times, speed, noise, dist_slack)
    results_lightweight = Dict{Int, Tuple{Vector{Float64},Vector{Float64}}}()
    for num_particles in num_particles_list
        ess_threshold = num_particles / 2
        #ess_threshold = 0.
        elapsed = Vector{Float64}(undef, num_reps)
        lmls = Vector{Float64}(undef, num_reps)
        for rep=1:num_reps
            start = time_ns()
            (traces, _, lml) = particle_filter(lightweight_hmm, model_args_rest, max_steps, 
                                               num_particles, ess_threshold,
                                               get_lightweight_hmm_obs; verbose=verbose)
            elapsed[rep] = Int(time_ns() - start) / 1e9
            println("num_particles: $num_particles, lml estimate: $lml")
            lmls[rep] = lml
        end
        results_lightweight[num_particles] = (lmls, elapsed)
    end


    ## particle filtering in lightweight hmm using external proposal ##

    println("lightweight hmm..")

    function get_lightweight_hmm_init()
        observations = DynamicAssignment()
        observations[(:x, 1)] = measured_xs[1]
        observations[(:y, 1)] = measured_ys[1]
        proposal_args = (speed, times, dist_slack)
        return (observations, proposal_args)
    end

    function get_lightweight_hmm_step(step::Int, trace)
        @assert step > 1
        observations = DynamicAssignment()
        observations[(:x, step)] = measured_xs[step]
        observations[(:y, step)] = measured_ys[step]
        prev_dist = get_assignment(trace)[(:dist, step-1)]
        proposal_args = (step, prev_dist, speed, times, dist_slack)
        return (observations, proposal_args)
    end

    model_args_rest = (path, distances_from_start, times, speed, noise, dist_slack)
    results_lightweight_external = Dict{Int, Tuple{Vector{Float64},Vector{Float64}}}()
    for num_particles in num_particles_list
        ess_threshold = num_particles / 2
        #ess_threshold = 0.
        elapsed = Vector{Float64}(undef, num_reps)
        lmls = Vector{Float64}(undef, num_reps)
        for rep=1:num_reps
            start = time_ns()
            (traces, _, lml) = particle_filter(lightweight_hmm, model_args_rest, max_steps,
                                               num_particles, ess_threshold,
                                               get_lightweight_hmm_init,
                                               get_lightweight_hmm_step,
                                               lightweight_init_proposal,
                                               lightweight_step_proposal;verbose=verbose)
            elapsed[rep] = Int(time_ns() - start) / 1e9
            println("num_particles: $num_particles, lml estimate: $lml")
            lmls[rep] = lml
        end
        results_lightweight_external[num_particles] = (lmls, elapsed)
    end


    ## particle filtering in compiled hmm using internal proposal ##

    println("compiled hmm..")

    function get_compiled_hmm_obs(step::Int)
        assignment = DynamicAssignment()
        assignment[step => :x] = measured_xs[step]
        assignment[step => :y] = measured_ys[step]
        return assignment
    end

    kernel_params = KernelParams(times, path, distances_from_start, speed, dist_slack, noise)
    model_args_rest = (KernelState(NaN, Point(NaN, NaN)), kernel_params)
    results_compiled = Dict{Int, Tuple{Vector{Float64},Vector{Float64}}}()
    for num_particles in num_particles_list
        ess_threshold = num_particles / 2
        #ess_threshold = 0.
        elapsed = Vector{Float64}(undef, num_reps)
        lmls = Vector{Float64}(undef, num_reps)
        for rep=1:num_reps
            start = time_ns()
            (traces, _, lml) = particle_filter(compiled_hmm, model_args_rest, max_steps,
                                                     num_particles, ess_threshold,
                                                     get_compiled_hmm_obs; verbose=verbose)
            elapsed[rep] = Int(time_ns() - start) / 1e9
            println("num_particles: $num_particles, lml estimate: $lml")
            lmls[rep] = lml
        end
        results_compiled[num_particles] = (lmls, elapsed)
    end

    # plot results

    figure(figsize=(8, 8))

    median_elapsed = [median(results_lightweight_external[num_particles][2]) for num_particles in num_particles_list]
    mean_lmls = [mean(results_lightweight_external[num_particles][1]) for num_particles in num_particles_list]
    plot(median_elapsed, mean_lmls, label="lightweight (external)", color="cyan")

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
