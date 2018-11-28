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
retdiff = add_julia_node!(builder, 
    () -> nothing, [], gensym(), Nothing)
set_retdiff_node!(builder, retdiff)
set_return_node!(builder, y)
datum_ir = build_ir(builder)

render_graph(datum_ir, "datum")

(datum_trace_defn, datum_trace_struct_name) = generate_trace_type_and_methods(datum_ir, :datum)
datum_defn = generate_generative_function(datum_ir, :datum, datum_trace_struct_name)
eval(datum_trace_defn)
eval(datum_defn)

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

builder = StaticIRBuilder()
xs = add_argument_node!(builder, :xs, Vector{Float64})
n = add_julia_node!(builder, (xs) -> length(xs), [xs], :n, Int)
c = add_constant_node!(builder, 1)
inlier_std = add_random_choice_node!(builder, gamma, [c, c], :inlier_std, :inlier_std, Float64)
outlier_std = add_random_choice_node!(builder, gamma, [c, c], :outlier_std, :outlier_std, Float64)
slope = add_random_choice_node!(builder, normal, [c, c], :slope, :slope, Float64)
intercept = add_random_choice_node!(builder, normal, [c, c], :intercept, :intercept, Float64)
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
retdiff = add_constant_node!(builder, nothing)
set_return_node!(builder, ys)
set_retdiff_node!(builder, retdiff)
model_ir = build_ir(builder)

render_graph(model_ir, "model")

(model_trace_defn, model_trace_struct_name) = generate_trace_type_and_methods(model_ir, :model)
model_defn = generate_generative_function(model_ir, :model, model_trace_struct_name)
eval(model_trace_defn)
eval(model_defn)

Gen.load_generated_functions()


exit()
