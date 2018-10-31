include("scenes.jl")
include("path_planner.jl")
include("model.jl")

import Random

function make_scene()
    scene = Scene(0, 100, 0, 100) 
    add!(scene, Tree(Point(30, 20)))
    add!(scene, Tree(Point(83, 80)))
    add!(scene, Tree(Point(80, 40)))
    wall_height = 30.
    walls = [
        Wall(Point(20., 40.), 1, 40., 2., wall_height)
        Wall(Point(60., 40.), 2, 40., 2., wall_height)
        Wall(Point(60. - 15., 80.), 1, 15. + 2., 2., wall_height)
        Wall(Point(20., 80.), 1, 15., 2., wall_height)
        Wall(Point(20., 40.), 2, 40., 2., wall_height)]
    for wall in walls
        add!(scene, wall)
    end
    return scene
end

const scene = make_scene()
const times = collect(range(0, stop=1, length=20))

function show_prior_samples()
    Random.seed!(0)
    figure(figsize=(32, 32))
    for i=1:15
        subplot(4, 4, i)
        ax = gca()
        trace = simulate(model, (scene, times))
        render(scene, trace, ax)
    end
    savefig("demo.png")
end
# generate_prior_samples()

@gen function start_proposal()
    @addr(uniform(0, 1), :start_x)
    @addr(uniform(0, 1), :start_y)
end

@gen function stop_proposal()
    @addr(uniform(0, 1), :stop_x)
    @addr(uniform(0, 1), :stop_y)
end

@gen function speed_proposal()
    @addr(uniform(0, 1), :speed)
end

@gen function noise_proposal()
    @addr(uniform(0, 1), :noise)
end

function inference(measurements::Vector{Point}, iters::Int)
    constraints = Assignment()
    for (i, pt) in enumerate(measurements)
        constraints[i => :x] = pt.x
        constraints[i => :y] = pt.y
    end

    (trace, _) = generate(model, (scene, times), constraints)

    for iter=1:iters
        println("iter: $iter")
        trace = mh(model, start_proposal, (), trace)
        trace = mh(model, stop_proposal, (), trace)
        trace = mh(model, speed_proposal, (), trace)
        trace = mh(model, noise_proposal, (), trace)
    end

    return trace
end

function experiment()

    # generate simulated ground truth
    Random.seed!(0)
    constraints = DynamicAssignment()
    constraints[:start_x] = 0.1
    constraints[:start_y] = 0.1
    constraints[:stop_x] = 0.5
    constraints[:stop_y] = 0.5
    constraints[:noise] = 0.1
    (trace, _) = generate(model, (scene, times), constraints)

    figure(figsize=(4, 4))
    ax = gca()
    render(scene, trace, ax)
    savefig("ground_truth.png")

    assignment = get_assignment(trace)
    measurements = [Point(assignment[i => :x], assignment[i => :y]) for i=1:length(times)]
    start = Point(assignment[:start_x], assignment[:start_y])
end

experiment()
