using Gen
import Random

using FunctionalCollections

function strip_lineinfo(expr::Expr)
    @assert !(expr.head == :line)
    new_args = []
    for arg in expr.args
        if (isa(arg, Expr) && arg.head == :line) || isa(arg, LineNumberNode)
        elseif isa(arg, Expr) && arg.head == :block
            stripped = strip_lineinfo(arg)
            append!(new_args, stripped.args)
        else
            push!(new_args, strip_lineinfo(arg))
        end
    end
    Expr(expr.head, new_args...)
end

function strip_lineinfo(expr)
    expr
end

#########
# model #
#########

import Distributions
import Gen: random, logpdf, get_static_argument_types
struct TwoNormals <: Distribution{Float64} end
const two_normals = TwoNormals()
function logpdf(::TwoNormals, x, mu, sigma1, sigma2)
    if sigma1 < 0 || sigma2 < 0
        return -Inf
    end
    l1 = Distributions.logpdf(Distributions.Normal(mu, sigma1), x) + log(0.5)
    l2 = Distributions.logpdf(Distributions.Normal(mu, sigma2), x) + log(0.5)
    m = max(l1, l2)
    m + log(exp(l1 - m) + exp(l2 - m))
end
function random(::TwoNormals, mu, sigma1, sigma2)
    mu + (rand() < 0.5 ? sigma1 : sigma2) * randn()
end
get_static_argument_types(::TwoNormals) = [:Float64, :Float64, :Float64]

struct Params
    prob_outlier::Float64
    inlier_std::Float64
    outlier_std::Float64
    slope::Float64
    intercept::Float64
end

@compiled @gen function datum(x::Float64, params::Params)
    is_outlier::Bool = @addr(bernoulli(params.prob_outlier), :z)
    std::Float64 = is_outlier ? params.inlier_std : params.outlier_std
    y::Float64 = @addr(normal(x * params.slope + params.intercept, std), :y)
    return y
end

data = plate(datum)
data_collapsed = plate(two_normals)

function compute_data_change(inlier_std_change, outlier_std_change, slope_change, intercept_change)
    if all([c !== nothing && (c == NoChange() || !c[1]) for c in [
            inlier_std_change, outlier_std_change, slope_change, intercept_change]])
        NoChange()
    else
        nothing
    end
end

@compiled @gen function model(xs::Vector{Float64})
    n::Int = length(xs)
    inlier_std::Float64 = @addr(Gen.gamma(1, 1), :inlier_std)
    outlier_std::Float64 = @addr(Gen.gamma(1, 1), :outlier_std)
    slope::Float64 = @addr(normal(0, 2), :slope)
    intercept::Float64 = @addr(normal(0, 2), :intercept)
    params::Params = Params(0.5, inlier_std, outlier_std, slope, intercept)
    inlier_std_change::Union{Tuple{Bool,Float64},Nothing} = @change(:inlier_std)
    outlier_std_change::Union{Tuple{Bool,Float64},Nothing} = @change(:outlier_std)
    slope_change::Union{Tuple{Bool,Float64},Nothing} = @change(:slope)
    intercept_change::Union{Tuple{Bool,Float64},Nothing} = @change(:intercept)
    change::Union{NoChange,Nothing} = compute_data_change(
        inlier_std_change, outlier_std_change, slope_change, intercept_change)
    ys::PersistentVector{Float64} = @addr(data(xs, fill(params, n)), :data, change)
    return ys
end

@compiled @gen function model_collapsed(xs::Vector{Float64})
    n::Int = length(xs)
    inlier_std::Float64 = @addr(Gen.gamma(1, 1), :inlier_std)
    outlier_std::Float64 = @addr(Gen.gamma(1, 1), :outlier_std)
    slope::Float64 = @addr(normal(0, 2), :slope)
    intercept::Float64 = @addr(normal(0, 2), :intercept)
    #inlier_std_change::Union{Tuple{Bool,Float64},Nothing} = @change(:inlier_std)
    #outlier_std_change::Union{Tuple{Bool,Float64},Nothing} = @change(:outlier_std)
    #slope_change::Union{Tuple{Bool,Float64},Nothing} = @change(:slope)
    #intercept_change::Union{Tuple{Bool,Float64},Nothing} = @change(:intercept)
    #change::Union{NoChange,Nothing} = compute_data_change(
        #inlier_std_change, outlier_std_change, slope_change, intercept_change)
    means::Vector{Float64} = broadcast(+, slope * xs, intercept)
    ys::PersistentVector{Float64} = @addr(data_collapsed(means, fill(inlier_std, n), fill(outlier_std, n)), :data)#, change)
    return ys
end


#######################
# inference operators #
#######################

@compiled @gen function slope_proposal()
    slope::Float64 = @read(:slope)
    @addr(normal(slope, 0.5), :slope)
end

@compiled @gen function intercept_proposal()
    intercept::Float64 = @read(:intercept)
    @addr(normal(intercept, 0.5), :intercept)
end

@compiled @gen function inlier_std_proposal()
    inlier_std::Float64 = @read(:inlier_std)
    @addr(normal(inlier_std, 0.5), :inlier_std)
end

@compiled @gen function outlier_std_proposal()
    outlier_std::Float64 = @read(:outlier_std)
    @addr(normal(outlier_std, 0.5), :outlier_std)
end

@compiled @gen function flip_z(z::Bool)
    @addr(bernoulli(z ? 0.0 : 1.0), :z)
end

data_proposal = at_dynamic(flip_z, Int)

@compiled @gen function is_outlier_proposal(i::Int)
    prev::Bool = @read(:data => i => :z)
    # TODO introduce shorthand @addr(flip_z(zs[i]), :data => i)
    @addr(data_proposal(i, (prev,)), :data) 
end

@compiled @gen function observe_datum(y::Float64)
    @addr(dirac(y), :y)
end

observe_data = plate(observe_datum)

@compiled @gen function observer(ys::Vector{Float64})
    @addr(observe_data(ys), :data)
end

@gen function observer_collapsed(ys::Vector{Float64})
    for (i, y) in enumerate(ys)
        @addr(dirac(y), :data => i)
    end
end

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
xs = collect(range(-5, stop=5, length=200))
ys = Float64[]
for (i, x) in enumerate(xs)
    if rand() < prob_outlier
        y = true_slope * x + true_intercept + randn() * true_inlier_noise
    else
        y = true_slope * x + true_intercept + randn() * true_outlier_noise
    end
    push!(ys, y)
end

##################
# run experiment #
##################

#println("\n######################################################################\n")
#println("simulate model:")
#println(strip_lineinfo(
    #Gen.codegen_simulate(typeof(model), Tuple{}, Nothing)))
#println("\n######################################################################\n")

#trace = simulate(model, (xs,))
#proposal_trace = simulate(slope_proposal, (), get_choices(trace))
#constraints = get_choices(proposal_trace)
#println("\n######################################################################\n")
#println("update model on constraints from slope_proposal:")
#println(strip_lineinfo(
    #Gen.codegen_update(typeof(model), Tuple{Vector{Float64}}, NoChange, typeof(trace), typeof(constraints), Nothing)))
#println("\n######################################################################\n")

function do_inference(n)
    observations = get_choices(simulate(observer_collapsed, (ys,)))

    # initial trace
    (trace, weight) = generate(model_collapsed, (xs,), observations)

    for i=1:n
    
        # steps on the parameters
        for j=1:5
            trace = mh(model_collapsed, slope_proposal, (), trace)
            trace = mh(model_collapsed, intercept_proposal, (), trace)
            trace = mh(model_collapsed, inlier_std_proposal, (), trace)
            trace = mh(model_collapsed, outlier_std_proposal, (), trace)
        end
   
        # step on the outliers (collapsed only)
        #for j=1:length(xs)
            #trace = mh(model_collapsed, is_outlier_proposal, (j,), trace)
        #end
    
        score = get_call_record(trace).score
    
        # print
        choices = get_choices(trace)
        slope = choices[:slope]
        intercept = choices[:intercept]
        inlier_std = choices[:inlier_std]
        outlier_std = choices[:outlier_std]
        println("score: $score, slope: $slope, intercept: $intercept, inlier_std: $inlier_std, outlier_std: $outlier_std")
    end
end

@time do_inference(100)
@time do_inference(10)
