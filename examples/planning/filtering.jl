using Gen
using PyPlot

include("scenes.jl")
include("path_planner.jl")

# depends on: scenes.jl, path_planner.jl

@gen function model(step::Int, path::Path, times::Vector{Float64}, speed::Float64,
                    noise::Float64, dist_slack::Float64)

    # walk path
	locations = Vector{Point}(undef, length(times))
 	distances_from_start = compute_distances_from_start(path)
	dist = @addr(normal(speed * times[1], dist_slack), (:dist, 1))
	locations[1] = walk_path(path, distances_from_start, dist)
	for t=2:step
		dist = @addr(normal(dist + speed * (times[t] - times[t-1]), dist_slack), (:dist, t))
		locations[t] = walk_path(path, distances_from_start, dist)
	end

    # generate noisy observations
    for i=1:step
    #for (i, point) in enumerate(locations)
        point = locations[i]
        @addr(normal(point.x, noise), (i, :x))
        @addr(normal(point.y, noise), (i, :y))
    end

    return locations
end

using Printf: @sprintf
import Random

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

function render(scene::Scene, start::Point, stop::Point, maybe_path::Nullable{Path},
                times::Vector{Float64}, speed::Float64,
                noise::Float64, dist_slack::Float64, trace, ax;
                show_measurements=true,
                show_start=true, show_stop=true,
                show_path=true, show_noise=true,
                start_alpha=1., stop_alpha=1., path_alpha=1.)

    locations = get_call_record(trace).retval
    assignment = get_assignment(trace)
    render(scene, ax)
    sca(ax)
    if show_start
        scatter([start.x], [start.y], color="blue", s=100, alpha=start_alpha)
    end
    if show_stop
        scatter([stop.x], [stop.y], color="red", s=100, alpha=stop_alpha)
    end
    if !isnull(maybe_path) && show_path
        path = get(maybe_path)
        for i=1:length(path.points)-1
            prev = path.points[i]
            next = path.points[i+1]
            plot([prev.x, next.x], [prev.y, next.y], color="black", alpha=0.5, linewidth=5)
        end
    end

    # plot locations with measurement noise around them
    scatter([pt.x for pt in locations], [pt.y for pt in locations],
        color="orange", alpha=path_alpha, s=25)
    if show_noise
        for pt in locations
            circle = patches[:Circle]((pt.x, pt.y), noise, facecolor="purple", alpha=0.2)
            ax[:add_patch](circle)
        end
    end
    
    # plot measured locations
    if show_measurements
        measured_xs = [assignment[(i, :x)] for i=1:length(locations)]
        measured_ys = [assignment[(i, :y)] for i=1:length(locations)]
        scatter(measured_xs, measured_ys, marker="x", color="black", alpha=1., s=25)
    end
end

function show_prior_samples()
    Random.seed!(0)

    # generate a path
    start_x = 0.1
    start_y = 0.1
    stop_x = 0.5
    stop_y = 0.5
    speed = 0.5
    noise = 0.01
    dist_slack = 0.1
    start = Point(start_x, start_y)
    stop = Point(stop_x, stop_y)
    maybe_path = plan_path(start, stop, scene, PlannerParams(300, 3.0, 2000, 1.))
    @assert !isnull(maybe_path)
    path = get(maybe_path)

    figure(figsize=(32, 32))
    for i=1:15
        subplot(4, 4, i)
        ax = gca()
        trace = simulate(model, (length(times), path, times, speed, noise, dist_slack))
        render(scene, start, stop, maybe_path, times, speed, noise, dist_slack, trace, ax)
    end
    savefig("filtering.png")
end
#show_prior_samples()

function experiment()

    Random.seed!(0)

    # generate a path
    start_x = 0.1
    start_y = 0.1
    stop_x = 0.5
    stop_y = 0.5
    speed = 0.5
    noise = 0.01
    dist_slack = 0.3
    start = Point(start_x, start_y)
    stop = Point(stop_x, stop_y)
    maybe_path = plan_path(start, stop, scene, PlannerParams(300, 3.0, 2000, 1.))
    @assert !isnull(maybe_path)
    path = get(maybe_path)

    # generate ground truth locations and observations
    trace = simulate(model, (length(times), path, times, speed, noise, dist_slack))
    locations = get_call_record(trace).retval
    model_args_rest = (path, times, speed, noise, dist_slack)
    assignment = get_assignment(trace)
    measured_xs = [assignment[(i, :x)] for i=1:length(locations)]
    measured_ys = [assignment[(i, :y)] for i=1:length(locations)]
 
    function get_observations(step::Int)
        assignment = DynamicAssignment()
        assignment[(step, :x)] = measured_xs[step]
        assignment[(step, :y)] = measured_ys[step]
        return assignment
    end

    for num_particles in [100]#[10, 100, 1000, 1000]
        ess_threshold = num_particles / 2
        for rep=1:5
            @time (traces, _, lml) = particle_filter(model, model_args_rest, length(times),
                                    num_particles, ess_threshold, get_observations)
            println("num_particles: $num_particles, lml estimate: $lml")
        end
    end
end

experiment()
