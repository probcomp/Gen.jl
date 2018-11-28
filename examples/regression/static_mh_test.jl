import Random
using Gen
using Gen: generate_trace_type_and_methods, generate_generative_function
using FunctionalCollections: PersistentVector

include("/home/marcoct/dev/Gen/src/static_dsl/render_ir.jl")

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
x = add_argument_node!(builder, :x, Float64)
params = add_argument_node!(builder, :params, Params)
prob_outlier = add_julia_node!(builder,
    (params) -> params.prob_outlier,
    [params], gensym(), Float64)
is_outlier = add_random_choice_node!(builder, bernoulli, [prob_outlier], :z, :is_outlier, Bool)
std = add_julia_node!(builder,
    (is_outlier, params) -> is_outlier ? params.inlier_std : params.outlier_std,
    [is_outlier, params], :std, Float64)
y_mean = add_julia_node!(builder, 
    (x, params) -> x * params.slope + params.intercept,
    [x, params], gensym(), Float64)
y = add_random_choice_node!(builder, normal, [y_mean, std], :y, :y, Float64)
received_argdiff = add_received_argdiff_node!(builder, :argdiff, Nothing)
set_return_node!(builder, y)
ir = build_ir(builder)

render_graph(ir, "datum")

eval(generate_generative_function(ir, :datum))

######################
# model as Static IR #
######################

data = Map(datum)

function compute_argdiff(inlier_std_diff, outlier_std_diff, slope_diff, intercept_diff)
    if all([c == NoChoiceDiff() for c in [
            inlier_std_diff, outlier_std_diff, slope_diff, intercept_diff]])
        noargdiff
        @assert false
    else
        unknownargdiff
    end
end

# TODO allow constants to be placed inline as arguments (nodes for them will be
# created behind the secenes)

builder = StaticIRBuilder()
xs = add_argument_node!(builder, :xs, Vector{Float64})
n = add_julia_node!(builder, (xs) -> length(xs), [xs], :n, Int)
zero = add_constant_node!(builder, 0.)
one = add_constant_node!(builder, 1.)
two = add_constant_node!(builder, 2.)
inlier_std = add_random_choice_node!(builder, gamma, [one, one], :inlier_std, :inlier_std, Float64)
outlier_std = add_random_choice_node!(builder, gamma, [one, one], :outlier_std, :outlier_std, Float64)
slope = add_random_choice_node!(builder, normal, [zero, two], :slope, :slope, Float64)
intercept = add_random_choice_node!(builder, normal, [zero, two], :intercept, :intercept, Float64)
params = add_julia_node!(builder,
    (inlier_std, outlier_std, slope, intercept) -> Params(0.5, inlier_std, outlier_std, slope, intercept),
    [inlier_std, outlier_std, slope, intercept], :params, Params)
inlier_std_diff = add_choicediff_node!(builder, :inlier_std, :inlier_std_diff, Any)
outlier_std_diff = add_choicediff_node!(builder, :outlier_std, :outlier_std_diff, Any)
slope_diff = add_choicediff_node!(builder, :slope, :slope_diff, Any)
intercept_diff = add_choicediff_node!(builder, :intercept, :intercept_diff, Any)
data_argdiff = add_diff_julia_node!(builder,
    (d1, d2, d3, d4) -> compute_argdiff(d1, d2, d3, d4),
    [inlier_std_diff, outlier_std_diff, slope_diff, intercept_diff],
    :argdiff, Any)
filled = add_julia_node!(builder,
    (params, n) -> fill(params, n), [params, n],
    gensym(), Vector{Params})
ys = add_gen_fn_call_node!(builder, data, [xs, filled], :data, data_argdiff, :ys, PersistentVector{Float64})
received_argdiff = add_received_argdiff_node!(builder, :argdiff, Nothing)
set_return_node!(builder, ys)
ir = build_ir(builder)

render_graph(ir, "model")

eval(generate_generative_function(ir, :model))

#######################
# inference operators #
#######################

builder = StaticIRBuilder()
prev = add_argument_node!(builder, :prev, Any)
prev_slope = add_julia_node!(builder,
    (prev) -> get_assignment(prev)[:slope],
    [prev], :slope, Float64)
c = add_constant_node!(builder, 0.5)
add_random_choice_node!(builder, normal, [prev_slope, c], :slope, gensym(), Float64)
ir = build_ir(builder)
eval(generate_generative_function(ir, :slope_proposal))

builder = StaticIRBuilder()
prev = add_argument_node!(builder, :prev, Any)
prev_intercept = add_julia_node!(builder,
    (prev) -> get_assignment(prev)[:intercept],
    [prev], :intercept, Float64)
c = add_constant_node!(builder, 0.5)
add_random_choice_node!(builder, normal, [prev_intercept, c], :intercept, gensym(), Float64)
ir = build_ir(builder)
eval(generate_generative_function(ir, :intercept_proposal))

builder = StaticIRBuilder()
prev = add_argument_node!(builder, :prev, Any)
prev_inlier_std = add_julia_node!(builder,
    (prev) -> get_assignment(prev)[:inlier_std],
    [prev], :inlier_std, Float64)
c = add_constant_node!(builder, 0.5)
add_random_choice_node!(builder, normal, [prev_inlier_std, c], :inlier_std, gensym(), Float64)
ir = build_ir(builder)
eval(generate_generative_function(ir, :inlier_std_proposal))

builder = StaticIRBuilder()
prev = add_argument_node!(builder, :prev, Any)
prev_outlier_std = add_julia_node!(builder,
    (prev) -> get_assignment(prev)[:outlier_std],
    [prev], :outlier_std, Float64)
c = add_constant_node!(builder, 0.5)
add_random_choice_node!(builder, normal, [prev_outlier_std, c], :outlier_std, gensym(), Float64)
ir = build_ir(builder)
eval(generate_generative_function(ir, :outlier_std_proposal))

@gen function is_outlier_proposal(prev, i::Int)
	prev_z::Bool = get_assignment(prev)[:data => i => :z]
    @addr(bernoulli(prev_z ? 0.0 : 1.0), :data => i => :z)
end

Gen.load_generated_functions()

#####################
# generate data set #
#####################

Random.seed!(1)

prob_outlier = 0.5
true_inlier_noise = 0.5
true_outlier_noise = 0.5#5.0
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
#observations[:inlier_std] = 0.5
#observations[:outlier_std] = 0.5

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
        #for j=1:length(xs)
            #trace = mh(model, is_outlier_proposal, (j,), trace)
        #end

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

@time (inlier_std, outlier_std, slope, intercept) = do_inference(100)
@time (inlier_std, outlier_std, slope, intercept) = do_inference(100)
