using GenLite
Gen = GenLite

############
# modeling #
############

include("scene.jl")
include("rrt.jl")
include("path_planner.jl")

const dim = 100.

function make_static_objects()
    scene = Scene(0, 100, 0, 100) # the scene spans the square [0, 100] x [0, 100]
    add!(scene, Tree(Point(30, 20))) # place a tree at x=30, y=20
    add!(scene, Tree(Point(83, 80)))
    add!(scene, Tree(Point(80, 40)))
    wall_height = 30.
    walls = [
        Wall(Point(20., 40.), 1, 40., 2., wall_height)
        Wall(Point(60., 40.), 2, 40., 2., wall_height)
        Wall(Point(60.-15., 80.), 1, 15. + 2., 2., wall_height)
        Wall(Point(20., 80.), 1, 15., 2., wall_height)
        Wall(Point(20., 40.), 2, 40., 2., wall_height)]
    for wall in walls
        add!(scene, wall)
    end
    return scene
end

function compute_distances_from_start(path::Vector{Point})
    num_path_steps = length(path)
    distances_from_start = Vector{Float64}(num_path_steps)
    distances_from_start[1] = 0.
    for i=2:num_path_steps
        distances_from_start[i] = distances_from_start[i-1] + dist(path[i], path[i-1])
    end
    return distances_from_start
end

function walk_path(path::Vector{Point}, distances_from_start::Vector{Float64}, dist::Float64)
    if dist <= 0.
        return path[1]
    end
    if dist >= distances_from_start[end]
        return path[end]
    end
    # dist > 0 and dist < dist-to-last-point
    for path_point_index=1:length(distances_from_start)
        if dist < distances_from_start[path_point_index]
            break
        end
    end
    @assert dist < distances_from_start[path_point_index]
    path_point_index -= 1
    @assert path_point_index > 0
    dist_from_path_point = dist - distances_from_start[path_point_index]
    dist_between_points = distances_from_start[path_point_index + 1] - distances_from_start[path_point_index]
    fraction_next = dist_from_path_point / dist_between_points
    point = fraction_next * path[path_point_index + 1] + (1. - fraction_next) * path[path_point_index]
    return point
end

const planner_params = PlannerParams(100, 10.0, 100, 1.)
const scene = make_static_objects()

model = @gen function (timestamps, start)

    num_waypoints = @addr(poisson(3.), "num")
    path = Point[start]
    failed = false
    prev_waypoint = start
    for i=1:num_waypoints
        x = @addr(uniform_continuous(0., dim), "wp-$i-x")
        y = @addr(uniform_continuous(0., dim), "wp-$i-y")
        if !failed
            waypoint = Point(x, y)
            sub_path = plan_path(prev_waypoint, waypoint, scene, planner_params)
            if isnull(sub_path)
                # the path terminates at the waypoint where planning first failed
                failed = true
            else
                append!(path, get(sub_path).points[2:end])
                prev_waypoint = waypoint
            end
        end
    end
    speed = 2.
    noise = 1.
    distances_from_start = compute_distances_from_start(path)
    for (i, timestamp) in enumerate(timestamps)
        dist = speed * timestamp
        point = walk_path(path, distances_from_start, dist)
        @addr(normal(point.x, noise), "obs-$i-x")
        @addr(normal(point.y, noise), "obs-$i-y")
    end
end


##################
# rendering code #
##################

using PyCall
@pyimport matplotlib.pyplot as plt
@pyimport matplotlib.patches as patches

function render(trace, start, num_obs; first_obs=1)
    ax = plt.gca()
    render(scene, ax)
    num = trace["num"]
    waypoint_xs = Float64[trace["wp-$i-x"] for i=1:num]
    waypoint_ys = Float64[trace["wp-$i-y"] for i=1:num]
    plt.scatter(waypoint_xs, waypoint_ys, color="red", s=50, alpha=0.5)
    plt.scatter([start.x], [start.y], color="blue", s=100)
    obs_xs = Float64[trace["obs-$i-x"] for i=first_obs:num_obs]
    obs_ys = Float64[trace["obs-$i-y"] for i=first_obs:num_obs]
    plt.scatter(obs_xs, obs_ys, color="purple", s=5, alpha=0.3)
    ax[:set_xlim](0, dim)
    ax[:set_ylim](0, dim)
end

function render(polygon::Polygon, ax)
    num_sides = length(polygon.vertices)
    vertices = Matrix{Float64}(num_sides, 2)
    for (i, pt) in enumerate(polygon.vertices)
        vertices[i,:] = [pt.x, pt.y]
    end
    poly = patches.Polygon(vertices, true, facecolor="black")
    ax[:add_patch](poly)
end

render(wall::Wall, ax) = render(wall.poly, ax)
render(tree::Tree, ax) = render(tree.poly, ax)

function render(scene::Scene, ax)
    for obstacle in scene.obstacles
        render(obstacle, ax)
    end
end


#######################
# inference operators #
#######################

waypoint_wiggle = @gen function (i, std)
    @addr(normal(@read("wp-$i-x"), std), "wp-$i-x")
    @addr(normal(@read("wp-$i-y"), std), "wp-$i-y")
end

new_waypoint = @gen function (observations::Matrix{Float64}, std)
    if rand() < 0.5
        # propose it from a normal distribution around an observation
        (n_obs, _) = size(observations)
        obs_idx = rand(uniform_discrete, (1, n_obs))
        (x, y) = observations[obs_idx,:]
        @addr(normal(x, std), "new-wp-x")
        @addr(normal(y, std), "new-wp-y")
    else
        # propose uniformly from the grid
        @addr(uniform_continuous(0., dim), "new-wp-x")
        @addr(uniform_continuous(0., dim), "new-wp-y")
    end

end

delete = @bijective function (waypoint_idx::Int, num_obs::Int)

    # decrement the number of waypoints
    num = @read((MODEL, "num"))
    @write(num - 1, (MODEL, "num"))

    # move the waypoint to the reverse proposal
    @copy((MODEL, "wp-$waypoint_idx-x"), (PROP, "new-wp-x"))
    @copy((MODEL, "wp-$waypoint_idx-y"), (PROP, "new-wp-y"))

    # copy all waypoints before waypoint_idx
    for i=1:waypoint_idx-1
        @copy((MODEL, "wp-$i-x"), (MODEL, "wp-$i-x"))
        @copy((MODEL, "wp-$i-y"), (MODEL, "wp-$i-y"))
    end

    # shift all waypoints after waypoint_idx to the left by one
    for i=waypoint_idx+1:num
        @copy((MODEL, "wp-$i-x"), (MODEL, "wp-$(i-1)-x"))
        @copy((MODEL, "wp-$i-y"), (MODEL, "wp-$(i-1)-y"))
    end

    # copy all observations
    for i=1:num_obs
        @copy((MODEL, "obs-$i-x"), (MODEL, "obs-$i-x"))
        @copy((MODEL, "obs-$i-y"), (MODEL, "obs-$i-y"))
    end
end

add = @bijective function (waypoint_idx::Int, num_obs::Int)

    # increment the number of waypoints
    num = @read((MODEL, "num"))
    @write(num + 1, (MODEL, "num"))

    # add the new waypoint
    @copy((PROP, "new-wp-x"), (MODEL, "wp-$waypoint_idx-x"))
    @copy((PROP, "new-wp-y"), (MODEL, "wp-$waypoint_idx-y"))

    # copy all waypoints before waypoint_idx
    for i=1:waypoint_idx-1
        @copy((MODEL, "wp-$i-x"), (MODEL, "wp-$i-x"))
        @copy((MODEL, "wp-$i-y"), (MODEL, "wp-$i-y"))
    end

    # shift all waypoints after and including waypoint_idx to the right by one
    for i=waypoint_idx:num
        @copy((MODEL, "wp-$i-x"), (MODEL, "wp-$(i+1)-x"))
        @copy((MODEL, "wp-$i-y"), (MODEL, "wp-$(i+1)-y"))
    end

    # copy all observations
    for i=1:num_obs
        @copy((MODEL, "obs-$i-x"), (MODEL, "obs-$i-x"))
        @copy((MODEL, "obs-$i-y"), (MODEL, "obs-$i-y"))
    end
end

add_transform = TraceTransform(add, delete)
delete_transform = inverse(add_transform)

function predict_all(cur_idx::Int, trace, score, all_timestamps)
    # fill in additional observations using forward simulation
    prev_args = (all_timestamps[1:cur_idx], start)
    new_args = (all_timestamps, start)
    (new_trace, _, _) = predict(model, new_args, trace)
    n = length(all_timestamps)
    @assert haskey(new_trace, "obs-$n-x")
    @assert haskey(new_trace, "obs-$n-y")
    new_trace
end

function add_delete_move(trace, score, aux, timestamps, observations)
    num_obs = length(timestamps)
    @assert size(observations) == (length(timestamps), 2)
    std = 10.
    num = trace["num"]
    if num > 0 && rand() < 0.5
        waypoint_idx = rand(uniform_discrete, (1, num))

        # delete move
        (new_trace, new_score, new_aux, alpha) = mh_trans(
                model, (timestamps, start),
                (@gen function () end), (), # fwd
                new_waypoint, (observations, std),  # rev
                delete_transform, (waypoint_idx, num_obs),
                trace, score)
    else
        waypoint_idx = rand(uniform_discrete, (1, num+1))

        # add move
        (new_trace, new_score, new_aux, alpha) = mh_trans(
                model, (timestamps, start),
                new_waypoint, (observations, std), # fwd
                (@gen function () end), (), # rev
                add_transform, (waypoint_idx, num_obs),
                trace, score)
    end

    # probability of picking a given waypoint to delete is 1/N where N is the
    # current number of waypoints
    # 1, 2, 3, 4, 5, 6
    #       * (deleted)
    # 1, 2,    3, 4, 5
    # ^  ^     ^  ^  ^ (could add in 6 possible positions, including at end)
    # therefore, no correction is needed.
    (trace, score, aux) = mh_accept_reject(alpha, trace, score, aux, new_trace, new_score, new_aux)
    (trace, score, aux)
end

function mcmc_sweep(trace, score, aux, timestamps, observations::Matrix{Float64})

    # random walk sweep over existing waypoints
    for i=1:trace["num"]
        (new_trace, new_score, new_aux, alpha) = mh_custom(model, (timestamps, start),
                                                    waypoint_wiggle, (i, 1.),
                                                    trace, score)
        (trace, score, aux) = mh_accept_reject(alpha, trace, score, aux, new_trace, new_score, new_aux)
    end
    
    # add / delete move
    add_delete_move(trace, score, aux, timestamps, observations)
end


#############################
# obtain simulated data set #
#############################

srand(3)
all_timestamps = collect(linspace(1, 100, 100))
start = Point(10., 10.)

# render prior samples
plt.figure(figsize=(16,16))
traces = []
for i=1:16
    (trace, _, _) = simulate(model, (all_timestamps, start))
    push!(traces, trace)
    plt.subplot(4, 4, i)
    render(trace, start, length(all_timestamps))
end
plt.tight_layout()
plt.savefig("prior_samples.png")

# extract one of the simulated datasets for testing
all_observations = Matrix{Float64}(length(all_timestamps), 2)
trace = traces[5]
for i=1:length(all_timestamps)
    x = trace["obs-$i-x"]
    y = trace["obs-$i-y"]
    all_observations[i,:] = [x, y]
end

########################
# inference experiment #
########################

function make_observations(i::Int, x, y)
    trace = Trace()
    trace["obs-$i-x"] = x
    trace["obs-$i-y"] = y
    trace
end

function normalized_weights(log_weights)
    m = maximum(log_weights)
    denom = log(sum(exp.(log_weights - m))) + m
    exp.(log_weights- denom)
end

function compute_ess(normalized_weights)
    1. / sum((normalized_weights .^ 2))
end

num_replicates = 100#1000

traces = Trace[]
scores = Float64[]
auxs = Trace[]
log_weights = Float64[]
for rep=1:num_replicates
    (trace, score, aux, _) = simulate(model, (Float64[], start))
    push!(traces, trace)
    push!(scores, score)
    push!(auxs, aux)
    push!(log_weights, 0.)
end
for i=1:length(all_timestamps)
    println("step $i")
    timestamps = all_timestamps[1:i]
    observations = all_observations[1:i,:]
    prev_args = (all_timestamps[1:i-1], start)
    new_args = (timestamps, start)

    new_obs = make_observations(i, observations[i,1], observations[i,2])
    
    normalized = normalized_weights(log_weights)
    ess = compute_ess(normalized)

    # resample
    println("effective sample size: $ess")
    if ess < Float64(num_replicates)/2.
        println("RESAMPLE")
        log_weights = zeros(num_replicates)
        parents = Int[]
        for rep=1:num_replicates
            parent = rand(categorical, (normalized,))
            push!(parents, parent)
            traces[rep] = deepcopy(traces[parent])
            scores[rep] = scores[parent]
        end
        println("parents: $parents")
    end

    # extend samples using forward simulation
    for rep=1:num_replicates
        println("rep: $rep")
        (traces[rep], scores[rep], auxs[rep], log_weight, _) = smc(
            model, new_args, new_obs, traces[rep], scores[rep])
        log_weights[rep] += log_weight

        # do mcmc rejuvenation steps
        for j=1:10
            (traces[rep], scores[rep], auxs[rep]) = mcmc_sweep(traces[rep], scores[rep], auxs[rep], timestamps, observations)
        end
    end

    plt.figure()
    ax = plt.gca()
    render(scene, ax)
    for (rep, (trace, score)) in enumerate(zip(traces, scores))
        println("render: $rep")
        prediction = predict_all(i, trace, score, all_timestamps)
        render(prediction, start, length(all_timestamps); first_obs=i+1)
    end
    plt.scatter(observations[1:i,1], observations[1:i,2], color="black", s=20)
    plt.savefig("inferences-$i.png")
end

