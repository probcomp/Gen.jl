import Random
using Gen
using Gen: generate_trace_type_and_methods, generate_generative_function
using FunctionalCollections: PersistentVector

include("/home/marcoct/dev/Gen/src/static_ir/render_ir.jl")

struct Params
    prob_outlier::Float64
    inlier_std::Float64
    outlier_std::Float64
    slope::Float64
    intercept::Float64
end

######################
# datum as Static IR #
######################

# TODO when we add a Julia node, we should be able to add custom backpropagation code for it

builder = StaticIRBuilder()
x = add_argument_node!(builder, name=:x, typ=Float64)
params = add_argument_node!(builder, name=:params, typ=Params)
prob_outlier = add_julia_node!(builder,
    (params) -> params.prob_outlier,
    inputs=[params])
is_outlier = add_random_choice_node!(builder, bernoulli, inputs=[prob_outlier], addr=:z, name=:is_outlier, typ=Bool)
std = add_julia_node!(builder,
    (is_outlier, params) -> is_outlier ? params.inlier_std : params.outlier_std,
    inputs=[is_outlier, params], name=:std)
y_mean = add_julia_node!(builder, 
    (x, params) -> x * params.slope + params.intercept,
    inputs=[x, params])
y = add_random_choice_node!(builder, normal, inputs=[y_mean, std], addr=:y, name=:y, typ=Float64)
set_return_node!(builder, y)
ir = build_ir(builder)

#render_graph(ir, "datum")

eval(generate_generative_function(ir, :datum))

######################
# model as Static IR #
######################

data = Map(datum)

function compute_argdiff(inlier_std_diff, outlier_std_diff, slope_diff, intercept_diff)
    if all([c == NoChoiceDiff() for c in [
            inlier_std_diff, outlier_std_diff, slope_diff, intercept_diff]])
        noargdiff
    else
        unknownargdiff
    end
end

# TODO allow constants to be placed inline as arguments (nodes for them will be
# created behind the secenes)

builder = StaticIRBuilder()
xs = add_argument_node!(builder, name=:xs, typ=Vector{Float64})
n = add_julia_node!(builder, (xs) -> length(xs), inputs=[xs], name=:n, typ=Int)
zero = add_constant_node!(builder, 0.)
one = add_constant_node!(builder, 1.)
two = add_constant_node!(builder, 2.)
inlier_std = add_random_choice_node!(builder, gamma, inputs=[one, one], addr=:inlier_std, name=:inlier_std, typ=Float64)
outlier_std = add_random_choice_node!(builder, gamma, inputs=[one, one], addr=:outlier_std, name=:outlier_std, typ=Float64)
slope = add_random_choice_node!(builder, normal, inputs=[zero, two], addr=:slope, name=:slope, typ=Float64)
intercept = add_random_choice_node!(builder, normal, inputs=[zero, two], addr=:intercept, name=:intercept, typ=Float64)
params = add_julia_node!(builder,
    (inlier_std, outlier_std, slope, intercept) -> Params(0.5, inlier_std, outlier_std, slope, intercept),
    inputs=[inlier_std, outlier_std, slope, intercept], name=:params)
inlier_std_diff = add_choicediff_node!(builder, :inlier_std, name=:inlier_std_diff)
outlier_std_diff = add_choicediff_node!(builder, :outlier_std, name=:outlier_std_diff)
slope_diff = add_choicediff_node!(builder, :slope, name=:slope_diff)
intercept_diff = add_choicediff_node!(builder, :intercept, name=:intercept_diff)
data_argdiff = add_diff_julia_node!(builder,
    (d1, d2, d3, d4) -> compute_argdiff(d1, d2, d3, d4),
    inputs=[inlier_std_diff, outlier_std_diff, slope_diff, intercept_diff])
filled = add_julia_node!(builder,
    (params, n) -> fill(params, n), inputs=[params, n])
ys = add_gen_fn_call_node!(builder, data, inputs=[xs, filled], addr=:data, argdiff=data_argdiff, name=:ys, typ=PersistentVector{Float64})
set_return_node!(builder, ys)
ir = build_ir(builder)

#render_graph(ir, "model")

eval(generate_generative_function(ir, :model))

#######################
# inference operators #
#######################

# slope_proposal

builder = StaticIRBuilder()
prev = add_argument_node!(builder, name=:prev)
prev_slope = add_julia_node!(builder,
    (prev) -> get_assignment(prev)[:slope],
    inputs=[prev], name=:slope)
c = add_constant_node!(builder, 0.5)
add_random_choice_node!(builder, normal, inputs=[prev_slope, c], addr=:slope, typ=Float64)
ir = build_ir(builder)
eval(generate_generative_function(ir, :slope_proposal))

# intercept_proposal 

builder = StaticIRBuilder()
prev = add_argument_node!(builder, name=:prev)
prev_intercept = add_julia_node!(builder,
    (prev) -> get_assignment(prev)[:intercept],
    inputs=[prev], name=:intercept)
c = add_constant_node!(builder, 0.5)
add_random_choice_node!(builder, normal, inputs=[prev_intercept, c], addr=:intercept, typ=Float64)
ir = build_ir(builder)
eval(generate_generative_function(ir, :intercept_proposal))

# inlier_std

builder = StaticIRBuilder()
prev = add_argument_node!(builder, name=:prev)
prev_inlier_std = add_julia_node!(builder,
    (prev) -> get_assignment(prev)[:inlier_std],
    inputs=[prev], name=:inlier_std)
c = add_constant_node!(builder, 0.5)
add_random_choice_node!(builder, normal, inputs=[prev_inlier_std, c], addr=:inlier_std, typ=Float64)
ir = build_ir(builder)
eval(generate_generative_function(ir, :inlier_std_proposal))

# outlier_std

builder = StaticIRBuilder()
prev = add_argument_node!(builder, name=:prev)
prev_outlier_std = add_julia_node!(builder,
    (prev) -> get_assignment(prev)[:outlier_std],
    inputs=[prev], name=:outlier_std)
c = add_constant_node!(builder, 0.5)
add_random_choice_node!(builder, normal, inputs=[prev_outlier_std, c], addr=:outlier_std, typ=Float64)
ir = build_ir(builder)
eval(generate_generative_function(ir, :outlier_std_proposal))

# is_outlier_proposal

builder = StaticIRBuilder()
prob = add_argument_node!(builder, name=:prob, typ=Float64)
add_random_choice_node!(builder, bernoulli, inputs=[prob], addr=:z, typ=Bool)
eval(generate_generative_function(build_ir(builder), :flip_z))

data_proposal = at_dynamic(flip_z, Int)

builder = StaticIRBuilder()
prev = add_argument_node!(builder, name=:prev)
i = add_argument_node!(builder, name=:i, typ=Int)
prev_z = add_julia_node!(builder,
    (prev, i) -> get_assignment(prev)[:data => i => :z],
    inputs=[prev, i], name=:prev_z)
args = add_julia_node!(builder,
    (prev_z) -> (prev_z ? 0.0 : 1.0,),
    inputs=[prev_z])
add_gen_fn_call_node!(builder, data_proposal, inputs=[i, args], addr=:data)
eval(generate_generative_function(build_ir(builder), :is_outlier_proposal))

Gen.load_generated_functions()

#####################
# generate data set #
#####################

Random.seed!(1)

prob_outlier = 0.5
true_inlier_noise = 0.5
true_outlier_noise = 5.0
true_slope = -1
true_intercept = 2
xs = collect(range(-5, stop=5, length=1000))
ys = Float64[]
for (i, x) in enumerate(xs)
    if rand() < prob_outlier
        y = true_slope * x + true_intercept + randn() * true_inlier_noise
    else
        y = true_slope * x + true_intercept + randn() * true_outlier_noise
    end
    push!(ys, y)
end

######################
# inference programs #
######################

observations = DynamicAssignment()
for (i, y) in enumerate(ys)
    observations[:data => i => :y] = y
end

function show_code()

    (trace, weight) = generate(model, (xs,), observations)
    params = Params(0.5, 0., 0., 0., 0.)
    (datum_trace, weight) = generate(datum, (1.2, params), EmptyAssignment())
    
    # show code for update for is_outlier
    println("\n*** code for is_outlier update ***\n")
    proposed_trace = simulate(is_outlier_proposal, (trace, 1))
    constraints = StaticAssignment(get_assignment(proposed_trace))
    println(constraints)
    code = Gen.codegen_update(typeof(model), Tuple{Vector{Float64}}, NoArgDiff, typeof(trace), typeof(constraints))
    println(code)
    
    # show code for update for slope
    println("\n*** model code for slope update ***\n")
    proposed_trace = simulate(slope_proposal, (trace, 1))
    constraints = StaticAssignment(get_assignment(proposed_trace))
    println(constraints)
    code = Gen.codegen_update(typeof(model), Tuple{Vector{Float64}}, NoArgDiff, typeof(trace), typeof(constraints))
    println(code)
    
    # show code for update for slope
    # TODO
    println("\n*** datum code for slope update ***\n")
    code = Gen.codegen_update(typeof(datum), Tuple{Float64, Params}, UnknownArgDiff, typeof(datum_trace), EmptyAssignment)
    println(code)
    
    # show code for generate for slope proposal
    println("\n*** code for generate slope proposal ***\n")
    code = Gen.codegen_generate(typeof(slope_proposal), Tuple{typeof(trace)}, EmptyAssignment)
    println(code)
end
# show_code()

function do_inference(n)

    # initial trace
    (trace, weight) = generate(model, (xs,), observations)

    for i=1:n
    
        # steps on the parameters
        for j=1:5
            trace = mh(model, slope_proposal, (), trace)
            trace = mh(model, intercept_proposal, (), trace)
            trace = mh(model, inlier_std_proposal, (), trace)
            trace = mh(model, outlier_std_proposal, (), trace)
        end
   
        # step on the outliers
        for j=1:length(xs)
            trace = mh(model, is_outlier_proposal, (j,), trace)
        end

		assignment = get_assignment(trace)
		println((assignment[:inlier_std], assignment[:outlier_std], assignment[:slope], assignment[:intercept]))
    end

    assignment = get_assignment(trace)
    return (assignment[:inlier_std], assignment[:outlier_std], assignment[:slope], assignment[:intercept])
end


#################
# run inference #
#################

using Test

(inlier_std, outlier_std, slope, intercept) = do_inference(100)
max_std = max(inlier_std, outlier_std)
min_std = min(inlier_std, outlier_std)
#@test isapprox(min_std, 0.5, atol=1e-1)
#@test isapprox(max_std, 5.0, atol=1e-0)
#@test isapprox(slope, -1, atol=1e-1)
#@test isapprox(intercept, 2, atol=2e-1)

@time do_inference(100)

using Profile
Profile.clear()  # in case we have any previous profiling data
@profile do_inference(100)

li, lidict = Profile.retrieve()
using JLD
@save "static_mh_test.jlprof" li lidict

#using ProfileView
##ProfileView.view()
