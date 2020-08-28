using Nullables

# depends on: scenes.jl

###############
# Generic RRT #
###############

struct RRTNode{C,U}
    conf::C
    # the previous configuration and the control needed to produce this node's
    # configuration from the previous configuration may be null if it is the
    # root node
    parent::Nullable{RRTNode{C,U}}
    control::Nullable{U}
    cost_from_start::Float64 # cost (e.g. distance) of moving from start to us
end

mutable struct RRTTree{C,U}
    nodes::Array{RRTNode{C,U}, 1}
end

function RRTTree(root_conf::C, ::Type{U}) where {C,U}
    nodes = Array{RRTNode{C,U},1}()
    push!(nodes, RRTNode(root_conf,
        Nullable{RRTNode{C,U}}(), Nullable{U}(), 0.0))
    RRTTree(nodes)
end


function Base.show(io::IO, tree::RRTTree{C,U}) where {C,U}
    write(io, "RRTTree with $(length(tree.nodes)) nodes")
end

function add_node!(tree::RRTTree{C,U}, parent::RRTNode{C,U},
                   new_conf::C, control::U,
                   cost_from_start::Float64) where {C,U}
    node = RRTNode(new_conf,
        Nullable{RRTNode{C,U}}(parent), Nullable{U}(control),
        cost_from_start)
    push!(tree.nodes, node)
    return node
end

root(tree::RRTTree{C,U}) where {C,U} = tree.nodes[1]

abstract type RRTScheme{C,U} end

function nearest_neighbor(scheme::RRTScheme{C,U}, conf::C,
                               tree::RRTTree{C,U}) where {C,U}
    nearest::RRTNode{C,U} = root(tree)
    best_dist::Float64 = dist(nearest.conf, conf)
    for node::RRTNode{C,U} in tree.nodes
        d = dist(node.conf, conf)
        if d < best_dist
            best_dist = d
            nearest = node
        end
    end
    nearest
end

struct SelectControlResult{C,U}
    start_conf::C
    new_conf::C
    control::U
    failed::Bool # new_conf is undefined in this case
    cost::Float64 # cost of this control action (e.g. distance)
end

function rrt(scheme::RRTScheme{C,U}, init::C, iters::Int, dt::Float64) where {C,U}
    tree = RRTTree(init, U) # init is the root of tree
    for iter=1:iters
        rand_conf::C = random_config(scheme)
        near_node::RRTNode{C,U} = nearest_neighbor(scheme, rand_conf, tree)
        result = select_control(scheme, rand_conf, near_node.conf, dt)
        if !result.failed
            cost_from_start = near_node.cost_from_start + result.cost
            add_node!(tree, near_node, result.new_conf, result.control, cost_from_start)
        end
    end
    tree
end


##############################
# RRT for holonomic 2D point #
##############################

struct HolonomicPointRRTScheme <: RRTScheme{Point,Point}
    scene::Scene
end

function random_config(scheme::HolonomicPointRRTScheme)
    x = rand() * (scheme.scene.xmax - scheme.scene.xmin) + scheme.scene.xmin
    y = rand() * (scheme.scene.ymax - scheme.scene.ymin) + scheme.scene.ymin
    Point(x, y)
end

function select_control(scheme::HolonomicPointRRTScheme,
                        target_conf::Point, start_conf::Point, dt::Float64)

    dist_to_target = dist(start_conf, target_conf)
    diff = Point(target_conf.x - start_conf.x, target_conf.y - start_conf.y)
    distance_to_move = min(dt, dist_to_target)
    scale = distance_to_move / dist_to_target
    control = Point(scale * diff.x, scale * diff.y)

    obstacles = scheme.scene.obstacles

    # go in the direction of target_conf from start_conf
    new_conf = Point(start_conf.x + control.x, start_conf.y + control.y)

    # test the obstacles
    failed = false
    for obstacle in obstacles
        if intersects_path(obstacle, start_conf, new_conf)
            # NOTE: could do more intelligent things like backtrack until you succeed
            failed = true
            break
        end
    end
    cost = distance_to_move
    SelectControlResult(start_conf, new_conf, control, failed, cost)
end


################
# path planner #
################

struct PlannerParams
    rrt_iters::Int
    rrt_dt::Float64 # the maximum proposal distance
    refine_iters::Int
    refine_std::Float64
end

struct Path
    start::Point
    goal::Point
    points::Array{Point,1}
end

function concatenate(a::Path, b::Path)
    if a.goal.x != b.start.x || a.goal.y != b.start.y
        error("goal of first path muts be start of second path")
    end
    points = Array{Point,1}()
    for point in a.points
        push!(points, point)
    end
    for point in b.points[2:end]
        push!(points, point)
    end
    @assert points[1].x == a.start.x
    @assert points[1].y == a.start.y
    @assert points[end].x == b.goal.x
    @assert points[end].y == b.goal.y
    Path(a.start, b.goal, points)
end

"""
Remove intermediate nodes that are not necessary, so that refinement
optimization is a lower-dimensional optimization problem.
"""
function simplify_path(scene::Scene, original::Path)
    new_points = Array{Point,1}()
    push!(new_points, original.start)
    for i=2:length(original.points) - 1
        if !line_of_site(scene, new_points[end], original.points[i + 1])
            push!(new_points, original.points[i])
        end
    end
    @assert line_of_site(scene, new_points[end], original.goal)
    push!(new_points, original.goal)
    Path(original.start, original.goal, new_points)
end

"""
Optimize the path to minimize its length while avoiding obstacles in the scene.
"""
function refine_path(scene::Scene, original::Path, iters::Int, std::Float64)
    # do stochastic optimization
    new_points = deepcopy(original.points)
    num_interior_points = length(original.points) -2
    if num_interior_points == 0
        return original
    end
    for i=1:iters
        point_idx = 2 + (i % num_interior_points)
        @assert point_idx > 1 # not start
        @assert point_idx < length(original.points) # not goal
        prev_point = new_points[point_idx-1]
        point = new_points[point_idx]
        next_point = new_points[point_idx+1]
        adjusted = Point(point.x + randn() * std, point.y + randn() * std)
        cur_dist = dist(prev_point, point) + dist(point, next_point)
        ok_backward = line_of_site(scene, prev_point, adjusted)
        ok_forward = line_of_site(scene, adjusted, next_point)
        if ok_backward && ok_forward
            new_dist = dist(prev_point, adjusted) + dist(adjusted, next_point)
            if new_dist < cur_dist
                # accept the change
                new_points[point_idx] = adjusted
            end
        end
    end
    Path(original.start, original.goal, new_points)
end

function optimize_path(scene::Scene, original::Path, refine_iters::Int, refine_std::Float64)
    #simplified = simplify_path(scene, original) # TODO?
    refined = refine_path(scene, original, refine_iters, refine_std)
    refined
end

"""
Plan path from start to goal that avoids obstacles in the scene.
"""
function plan_path(start::Point, goal::Point, scene::Scene,
                   params::PlannerParams=PlannerParams(2000, 3.0, 10000, 1.))
    scheme = HolonomicPointRRTScheme(scene)
    tree = rrt(scheme, start, params.rrt_iters, params.rrt_dt)

    # find the best path along the tree to the goal, if one exists
    best_node = tree.nodes[1]
    min_cost = Inf
    path_found = false
    for node in tree.nodes
        # check for line-of-site to the goal
        clear_path = line_of_site(scene, node.conf, goal)
        cost = node.cost_from_start + (clear_path ? dist(node.conf, goal) : Inf)
        if cost < min_cost
            path_found = true
            best_node = node
            min_cost = cost
        end
    end

    local path::Nullable{Path}
    if path_found
        # extend the tree to the goal configuration
        control = Point(goal.x - best_node.conf.x, goal.y - best_node.conf.y)
        goal_node = add_node!(tree, best_node, goal, control, min_cost)
        points = Array{Point,1}()
        node::RRTNode{Point,Point} = goal_node
        push!(points, node.conf)
        # the path will contain the start and goal
        while !isnull(node.parent)
            node = get(node.parent)
            push!(points, node.conf)
        end
        @assert points[end] == start # the start point
        @assert points[1] == goal
        path = Nullable{Path}(Path(start, goal, reverse(points)))
    else
        path = Nullable{Path}()
    end

    local optimized_path::Nullable{Path}
    if path_found
        optimized_path = Nullable{Path}(optimize_path(scene, get(path),
                                                      params.refine_iters, params.refine_std))
    else
        optimized_path = Nullable{Path}()
    end
    return optimized_path
end

function compute_distances_from_start(path::Path)
    distances_from_start = Vector{Float64}(undef, length(path.points))
    distances_from_start[1] = 0.0
    for i=2:length(path.points)
        distances_from_start[i] = distances_from_start[i-1] + dist(path.points[i-1], path.points[i])
    end
    return distances_from_start
end

"""
Sample the location of a walk along a path at given time points, for a fixed speed.
"""
function walk_path(path::Path, speed::Float64, times::Array{Float64,1})
    distances_from_start = compute_distances_from_start(path)
    locations = Vector{Point}(undef, length(times))
    locations[1] = path.points[1]
    for (time_idx, t) in enumerate(times)
        if t < 0.0
            error("times must be positive")
        end
        desired_distance = t * speed
        used_up_time = false
        # NOTE: can be improved (iterate through path points along with times)
        for i=2:length(path.points)
            prev = path.points[i-1]
            cur = path.points[i]
            dist_to_prev = dist(prev, cur)
            if distances_from_start[i] >= desired_distance
                # we overshot, the location is between i-1 and i
                overshoot = distances_from_start[i] - desired_distance
                @assert overshoot <= dist_to_prev
                past_prev = dist_to_prev - overshoot
                frac = past_prev / dist_to_prev
                locations[time_idx] = Point(prev.x * (1. - frac) + cur.x * frac,
                                     prev.y * (1. - frac) + cur.y * frac)
                used_up_time = true
                break
            end
        end
        if !used_up_time
            # sit at the goal indefinitely
            locations[time_idx] = path.goal
        end
    end
    locations
end

function walk_path(path::Path, distances_from_start::Vector{Float64}, dist::Float64)
    if dist <= 0.
        return path.points[1]
    end
    if dist >= distances_from_start[end]
        return path.points[end]
    end
    # dist > 0 and dist < dist-to-last-point
    path_point_index = 0
    for i=1:length(distances_from_start)
        path_point_index += 1
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
    point::Point = (fraction_next * path.points[path_point_index + 1]
           + (1. - fraction_next) * path.points[path_point_index])
    return point
end

