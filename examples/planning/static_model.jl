using Gen
using PyPlot

# depends on: scenes.jl, path_planner.jl

function get_locations(maybe_path::Nullable{Path}, start::Point,
                       speed::Float64, times::Vector{Float64})
    if !isnull(maybe_path)
        path = get(maybe_path)
        return walk_path(path, speed, times)
    else
        return fill(start, length(times))
    end
end

@gen (static) function measurement(point::Point, noise::Float64)
    @trace(normal(point.x, noise), :x)
    @trace(normal(point.y, noise), :y)
end

measurements = Map(measurement)

@gen (static) function model(scene::Scene, times::Vector{Float64})

    # start point of the agent
    start_x::Float64 = @trace(uniform(0, 1), :start_x)
    start_y::Float64 = @trace(uniform(0, 1), :start_y)
    start = Point(start_x, start_y)

    # goal point of the agent
    stop_x::Float64 = @trace(uniform(0, 1), :stop_x)
    stop_y::Float64 = @trace(uniform(0, 1), :stop_y)
    stop = Point(stop_x, stop_y)

    # plan a path that avoids obstacles in the scene
    maybe_path = plan_path(start, stop, scene, PlannerParams(300, 3.0, 2000, 1.))

    # speed
    speed = @trace(uniform(0, 1), :speed)

    # walk path at constant speed
    locations = get_locations(maybe_path, start, speed, times)

    # generate noisy observations
    noise = @trace(uniform(0, 0.1), :noise)
    @trace(measurements(locations, fill(noise, length(times))), :measurements)

    ret = (start, stop, speed, noise, maybe_path, locations)
    return ret
end

function render(scene::Scene, trace, ax;
                show_measurements=true,
                show_start=true, show_stop=true,
                show_path=true, show_noise=true,
                start_alpha=1., stop_alpha=1., path_alpha=1.)
    (start, stop, speed, noise, maybe_path, locations) = get_call_record(trace).retval
    assignment = get_choices(trace)
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
        measured_xs = [assignment[:measurements => i => :x] for i=1:length(locations)]
        measured_ys = [assignment[:measurements => i => :y] for i=1:length(locations)]
        scatter(measured_xs, measured_ys, marker="x", color="black", alpha=1., s=25)
    end
end
