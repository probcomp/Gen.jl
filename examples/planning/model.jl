using Gen
using PyPlot

# depends on: scenes.jl, path_planner.jl

@gen function model(scene::Scene, times::Vector{Float64})

    # dimensions of the scene
    xmin = scene.xmin
    xmax = scene.xmax
    ymin = scene.ymin
    ymax = scene.ymax
    dimension = max(abs(xmax - xmin), abs(ymax - ymin))

    # start point of the agent
    start_x = xmin + (xmax - xmin) * @addr(uniform(0, 1), :start_x)
    start_y = ymin + (ymax - ymin) * @addr(uniform(0, 1), :start_y)
    start = Point(start_x, start_y)

    # stop point of the agent
    stop_x = xmin + (xmax - xmin) * @addr(uniform(0, 1), :stop_x)
    stop_y = xmin + (ymax - ymin) * @addr(uniform(0, 1), :stop_y)
    stop = Point(stop_x, stop_y)

    # plan a path that avoids obstacles in the scene
    maybe_path = plan_path(start, stop, scene, PlannerParams(2000, 3.0, 10000, 1.))
    
    # speed
    speed = dimension * @addr(uniform(0, 1), :speed)

    # walk path at constant speed
    if !isnull(maybe_path)
        path = get(maybe_path)
        locations = walk_path(path, speed, times)
    else
        locations = fill(start, length(times))
    end

    # generate noisy observations
    noise = dimension * 0.1 * @addr(uniform(0, 1), :noise)
    for (i, point) in enumerate(locations)
        @addr(normal(point.x, noise), i => :x)
        @addr(normal(point.y, noise), i => :y)
    end

    (start, stop, speed, noise, maybe_path, locations)
end

function render(scene::Scene, trace, ax)
    (start, stop, speed, noise, maybe_path, locations) = get_call_record(trace).retval
    assignment = get_assignment(trace)
    render(scene, ax)
    sca(ax)
    scatter([start.x], [start.y], color="blue", s=100)
    scatter([stop.x], [stop.y], color="red", s=100)
    if !isnull(maybe_path)
        path = get(maybe_path)
        for i=1:length(path.points)-1
            prev = path.points[i]
            next = path.points[i+1]
            plot([prev.x, next.x], [prev.y, next.y], color="black", alpha=0.5, linewidth=5)
        end
    end

    # plot locations with measurement noise around them
    scatter([pt.x for pt in locations], [pt.y for pt in locations], color="orange", alpha=1., s=25)
    for pt in locations
        circle = patches[:Circle]((pt.x, pt.y), noise, facecolor="purple", alpha=0.2)
        ax[:add_patch](circle)
    end
    
    # plot measured locations
    measured_xs = [assignment[i => :x] for i=1:length(locations)]
    measured_ys = [assignment[i => :y] for i=1:length(locations)]
    scatter(measured_xs, measured_ys, marker="x", color="black", alpha=1., s=25)
end



