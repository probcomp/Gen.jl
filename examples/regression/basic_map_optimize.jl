using Gen
import Random

using FunctionalCollections
using ReverseDiff

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

############################
# reverse mode AD for fill #
############################

function Base.fill(x::ReverseDiff.TrackedReal{V,D,O}, n::Integer) where {V,D,O}
    tp = ReverseDiff.tape(x)
    out = ReverseDiff.track(fill(ReverseDiff.value(x), n), V, tp)
    ReverseDiff.record!(tp, ReverseDiff.SpecialInstruction, fill, (x, n), out)
    return out
end

@noinline function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(fill)})
    x, n = instruction.input
    output = instruction.output
    ReverseDiff.istracked(x) && ReverseDiff.increment_deriv!(x, sum(ReverseDiff.deriv(output)))
    ReverseDiff.unseed!(output) 
    return nothing
end 

@noinline function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(fill)})
    x, n = instruction.input
    ReverseDiff.value!(instruction.output, fill(ReverseDiff.value(x), n))
    return nothing
end 

#########
# model #
#########

@compiled @gen function datum(x::Float64, @ad(inlier_std::Float64), @ad(outlier_std::Float64),
                                @ad(slope::Float64), @ad(intercept::Float64))
    is_outlier::Bool = @addr(bernoulli(0.5), :z)
    std::Float64 = is_outlier ? inlier_std : outlier_std
    y::Float64 = @addr(normal(x * slope + intercept, std), :y)
    return y
end

data = plate(datum)

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
    inlier_log_std::Float64 = @addr(normal(0, 2), :inlier_std)
    outlier_log_std::Float64 = @addr(normal(0, 2), :outlier_std)
    inlier_std::Float64 = exp(inlier_log_std)
    outlier_std::Float64 = exp(outlier_log_std)
    slope::Float64 = @addr(normal(0, 2), :slope)
    intercept::Float64 = @addr(normal(0, 2), :intercept)
    inlier_std_change::Union{Tuple{Bool,Float64},Nothing} = @change(:inlier_std)
    outlier_std_change::Union{Tuple{Bool,Float64},Nothing} = @change(:outlier_std)
    slope_change::Union{Tuple{Bool,Float64},Nothing} = @change(:slope)
    intercept_change::Union{Tuple{Bool,Float64},Nothing} = @change(:intercept)
    change::Union{NoChange,Nothing} = compute_data_change(
        inlier_std_change, outlier_std_change, slope_change, intercept_change)
    @addr(data(xs, fill(inlier_std, n), fill(outlier_std, n),
                                                   fill(slope, n), fill(intercept, n)),
                                          :data, change)
end

#######################
# inference operators #
#######################

@sel function slope_intercept_selector()
    @select(:slope)
    @select(:intercept)
end

@sel function std_selector()
    @select(:inlier_std)
    @select(:outlier_std)
end

@compiled @gen function flip_z(z::Bool)
    @addr(bernoulli(z ? 0.0 : 1.0), :z)
end

data_proposal = at_dynamic(flip_z, Int)

@compiled @gen function is_outlier_proposal(prev, i::Int)
    prev_z::Bool = get_choices(prev)[:data => i => :z]
    # TODO introduce shorthand @addr(flip_z(zs[i]), :data => i)
    @addr(data_proposal(i, (prev_z,)), :data) 
end

@compiled @gen function observe_datum(y::Float64)
    @addr(dirac(y), :y)
end

observe_data = plate(observe_datum)

@compiled @gen function observer(ys::Vector{Float64})
    @addr(observe_data(ys), :data)
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

trace = simulate(model, (xs,))
datum_trace = simulate(datum, (1., 2., 3., 4., 5.))
(slope_intercept_selection,) = Gen.select(slope_intercept_selector, (), get_choices(trace))
(std_selection,) = Gen.select(std_selector, (), get_choices(trace))
slope_intercept_static_sel = StaticAddressSet(slope_intercept_selection)
std_static_sel = StaticAddressSet(std_selection)

#println("\n######################################################################\n")
#println("backprop_trace(model, ...)")
#println(strip_lineinfo(
    #Gen.codegen_backprop_trace(typeof(model), typeof(trace), typeof(slope_intercept_static_sel), Nothing)))
#println("\n######################################################################\n")

#println("\n######################################################################\n")
#println("backprop_trace(datum, ...)")
#println(strip_lineinfo(
    #Gen.codegen_backprop_trace(typeof(datum), typeof(datum_trace),
           #Gen.EmptyAddressSet, Nothing)))
#println("\n######################################################################\n")


function do_inference(n)
    observations = get_choices(simulate(observer, (ys,)))
    
    # initial trace
    (trace, _) = generate(model, (xs,), observations)
    
    for i=1:n
        for j=1:5
            trace = map_optimize(model, slope_intercept_static_sel, trace, max_step_size=0.1, min_step_size=1e-10)
            trace = map_optimize(model, std_static_sel, trace, max_step_size=0.1, min_step_size=1e-10)
        end
    
        # step on the outliers
        for j=1:length(xs)
            trace = mh(model, is_outlier_proposal, (j,), trace)
        end
    
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
@time do_inference(100)
