using Gen
import Random
using FunctionalCollections
using ReverseDiff

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
    ys::PersistentVector{Float64} = @addr(data(xs, fill(inlier_std, n), fill(outlier_std, n),
                                                   fill(slope, n), fill(intercept, n)),
                                          :data, change)
    return ys
end


#######################
# inference operators #
#######################

@compiled @gen function mala_proposal(values, gradients, tau)
    std::Float64 = sqrt(2*tau)
    @addr(normal(values[:slope] + tau * gradients[:slope], std), :slope)
    @addr(normal(values[:intercept] + tau * gradients[:intercept], std), :intercept)
    @addr(normal(values[:inlier_std] + tau * gradients[:inlier_std], std), :inlier_std)
    @addr(normal(values[:outlier_std] + tau * gradients[:outlier_std], std), :outlier_std)
end

function make_mala_selection()
    set = DynamicAddressSet()
    for addr in [:slope, :intercept, :inlier_std, :outlier_std]
        Gen.push_leaf_node!(set, addr)
    end
    StaticAddressSet(set)
end

const mala_selection = make_mala_selection()

function mala_move(trace, tau::Float64)
    (_, values, gradients) = backprop_trace(model, trace, mala_selection, nothing)
    forward_trace = simulate(mala_proposal, (values, gradients, tau))
    forward_score = get_call_record(forward_trace).score
    constraints = get_choices(forward_trace)
    model_args = get_call_record(trace).args
    (new_trace, weight, discard) = update(
        model, model_args, NoChange(), trace, constraints)
    backward_trace = assess(mala_proposal, (values, gradients, tau), discard)
    backward_score = get_call_record(backward_trace).score
    alpha = weight - forward_score + backward_score
    if log(rand()) < alpha
        # accept
        return new_trace
    else
        # reject
        return trace
    end
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


function do_inference(n)
    observations = get_choices(simulate(observer, (ys,)))
    
    # initial trace
    (trace, _) = generate(model, (xs,), observations)
    
    for i=1:n
        trace = mala_move(trace, 0.001)
    
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
