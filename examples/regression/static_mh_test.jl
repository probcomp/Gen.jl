using Gen
using Gen: ArgumentNode, RandomChoiceNode, GenerativeFunctionCallNode, RegularNode, JuliaNode, StaticIRNode, StaticIR, generate_trace_type_and_methods, generate_generative_function
using Gen: DiffJuliaNode, ReceivedArgDiffNode, ChoiceDiffNode, CallDiffNode

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

x = ArgumentNode(:x, Float64)
params = ArgumentNode(:params, Params)
prob_outlier = JuliaNode(:(params.prob_outlier), Dict(:params => params), gensym(), Float64)
z = RandomChoiceNode(bernoulli, RegularNode[prob_outlier], :z, :is_outlier, Bool)
std = JuliaNode(:(is_outlier ? params.inlier_std : params.outlier_std),
          Dict(:is_outlier => z, :params => params), :std, Float64)
y_mean = JuliaNode(:(x * params.slope + params.intercept), Dict(:params => params, :x => x),
            gensym(), Float64)
y = RandomChoiceNode(normal, RegularNode[y_mean, std], :y, :y, Float64)
received_argdiff = ReceivedArgDiffNode(:argdiff, Nothing)
retdiff = JuliaNode(:(nothing), Dict{Symbol,Symbol}(), gensym(), Nothing)
nodes = StaticIRNode[x, params, prob_outlier, z, std, y_mean, y, received_argdiff, retdiff]
arg_nodes = [x, params]
choice_nodes = [z, y]
call_nodes = GenerativeFunctionCallNode[]
datum_ir = StaticIR(nodes, arg_nodes, choice_nodes, call_nodes, y, retdiff, received_argdiff)

render_graph(datum_ir, "datum.pdf")

(datum_trace_defn, datum_trace_struct_name) = generate_trace_type_and_methods(datum_ir, :datum)
datum_defn = generate_generative_function(datum_ir, :datum, datum_trace_struct_name)
eval(datum_trace_defn)
eval(datum_defn)

Gen.load_generated_functions()



exit()



@compiled @gen function datum(x::Float64, params::Params)
    is_outlier::Bool = @addr(bernoulli(params.prob_outlier), :z)
    std::Float64 = is_outlier ? params.inlier_std : params.outlier_std
    y::Float64 = @addr(normal(x * params.slope + params.intercept, std), :y)
    return y
end

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

@compiled @gen function model(xs::Vector{Float64})
    n::Int = length(xs)
    inlier_std::Float64 = @addr(gamma(1, 1), :inlier_std)
    outlier_std::Float64 = @addr(gamma(1, 1), :outlier_std)
    slope::Float64 = @addr(normal(0, 2), :slope)
    intercept::Float64 = @addr(normal(0, 2), :intercept)
    params::Params = Params(0.5, inlier_std, outlier_std, slope, intercept)
    inlier_std_diff::Union{PrevChoiceDiff{Float64},NoChoiceDiff} = @change(:inlier_std)
    outlier_std_diff::Union{PrevChoiceDiff{Float64},NoChoiceDiff} = @change(:outlier_std)
    slope_diff::Union{PrevChoiceDiff{Float64},NoChoiceDiff} = @change(:slope)
    intercept_diff::Union{PrevChoiceDiff{Float64},NoChoiceDiff} = @change(:intercept)
    argdiff::Union{NoArgDiff,UnknownArgDiff} = compute_argdiff(
        inlier_std_diff, outlier_std_diff, slope_diff, intercept_diff)
    ys::PersistentVector{Float64} = @addr(data(xs, fill(params, n)), :data, argdiff)
    return ys
end

