using Gen
using FunctionalCollections: PersistentVector
import Distributions

####################
# generative model #
####################

function x_mean(x_prev::Real, t::Int)
    (x_prev / 2.) + 25 * (x_prev / (1 + x_prev * x_prev)) + 8 * cos(1.2 * t)
end

y_mean(x::Real) = (x * x / 20.)

@gen function hmm(var_x, var_y, T::Int)
    xs = Vector{Float64}(undef, T)
    ys = Vector{Float64}(undef, T)
    xs[1] = @trace(normal(0, 5), :x => 1)
    ys[1] = @trace(normal(y_mean(xs[1]), sqrt(var_y)), :y => 1)
    for t=2:T
        xs[t] = @trace(normal(x_mean(xs[t-1], t), sqrt(var_x)), :x => t)
        ys[t] = @trace(normal(y_mean(xs[t]), sqrt(var_y)), :y => t)
    end
    return (xs, ys)
 end

struct State
    x::Float64
    y::Float64
end

struct Params
    var_x::Float64
    var_y::Float64
end

# the kernel accepts two special arguments (the time step t and the return
# value of the previous tieration, followed by the rest of the arguments,
# specified as an argument to unfold)
@gen (static) function kernel(t::Int, state::State, params::Params)
    x::Float64 = @trace(normal(t > 1 ? x_mean(state.x, t) : 0.,
                              t > 1 ? sqrt(params.var_x) : 5.), :x)
    y::Float64 = @trace(normal(y_mean(x), sqrt(params.var_y)), :y)
    ret::State = State(x, y)
    return ret
end

# arguments: (T::Int, init_state, common)
# i.e. the kernel needs to accept three arguments..
hmm2 = Unfold(kernel)

#################################
# sequential Monte Carlo in hmm #
#################################

#const num_particles = 200
#const ess = 100

# implement of SMC using forward simulation as the proposal
# it's okay to be specialized to the model 'hmm'

@gen function single_step_observer(t::Int, y::Float64)
    @trace(dirac(y), :y => t)
end

@gen (static) function obs_sub(y::Float64)
    @trace(dirac(y), :y)
end

single_step_observer2 = at_dynamic(obs_sub, Int)

#@gen (static) function single_step_observer2(t::Int, y::Float64)
    #@trace(at_dynamic(obs_sub,Int)(t, (y,)))
    #@trace(dirac(y), t => :y)
#end

function logsumexp(arr)
    min_arr = maximum(arr)
    min_arr + log(sum(exp.(arr .- min_arr)))
end

function effective_sample_size(log_weights::Vector{Float64})
    # assumes weights are normalized
    log_ess = -logsumexp(2. * log_weights)
    exp(log_ess)
end

# TODO need to implement generate and extend for the unfold generator type
# it should have a custom argchange argument that is indicates whether only the
# time length argument has changed, which will allow extend to only visit new
# time steps

function smc(var_x, var_y, T::Int, N, ess_threshold, ys::AbstractArray{Float64,1})
    log_unnormalized_weights = Vector{Float64}(undef, N)
    log_ml_estimate = 0.
    obs = get_choices(simulate(single_step_observer2, (1, (ys[1],))))
    traces = Vector{Gen.get_trace_type(hmm2)}(undef, N)
    next_traces = Vector{Gen.get_trace_type(hmm2)}(undef, N)
    args = (1, State(NaN, NaN), Params(var_x, var_y))
    for i=1:N
        (traces[i], log_unnormalized_weights[i]) = generate(hmm2, args, obs)
    end
    num_resamples = 0
    args_change = UnfoldCustomArgDiff(true, false, false)
    for t=2:T
        #println(t)
        log_total_weight = logsumexp(log_unnormalized_weights)
        log_normalized_weights = log_unnormalized_weights .- log_total_weight
        if effective_sample_size(log_normalized_weights) < ess_threshold
            weights = exp.(log_normalized_weights)
            parents = rand(Distributions.Categorical(weights / sum(weights)), N)
            #parents = map((i) -> categorical(weights / sum(weights)), 1:N)
            log_ml_estimate += log_total_weight - log(N)
            log_unnormalized_weights = zeros(N)
            num_resamples += 1
        else
            parents = 1:N
        end
        obs = get_choices(simulate(single_step_observer2, (t, (ys[t],))))
        args = (t, State(NaN, NaN), Params(var_x, var_y))
        for i=1:N
            parent = parents[i]
            (next_traces[i], weight) = extend(trace[i], hmm2, args, args_change, obs)
            log_unnormalized_weights[i] += weight
        end
        tmp = traces
        traces = next_traces
        next_traces = tmp
    end
    log_total_weight = logsumexp(log_unnormalized_weights)
    log_normalized_weights = log_unnormalized_weights .- log_total_weight
    log_ml_estimate += log_total_weight - log(N)
    log_ml_estimate # just return the log ML estimate
end


##########################################
# collapsed generative model (handcoded) #
##########################################

using Gen: VectorDistTrace, VectorDistTraceChoiceMap
import Gen: get_call_record, has_choices, get_choices, simulate, assess

struct CollapsedHMMTrace
    vector::VectorDistTrace{Float64}
end


get_call_record(trace::CollapsedHMMTrace) = trace.vector.call
has_choices(trace::CollapsedHMMTrace) = length(trace.vector.call.retval) > 0
get_choices(trace::CollapsedHMMTrace) = CollapsedHMMChoiceMap(get_choices(trace.vector))

struct CollapsedHMMChoiceMap <: ChoiceMap
    y_assignment::VectorDistTraceChoiceMap
end

# addrs are: :y => i

Base.isempty(assignment::CollapsedHMMChoiceMap) = (length(assignment.trace.call.retval) > 0)
Gen.get_address_schema(::Type{CollapsedHMMChoiceMap}) = DynamicAddressSchema()
Gen.has_internal_node(assignment::CollapsedHMMChoiceMap, addr) = (addr == :y)
function Gen.get_internal_node(assignment::CollapsedHMMChoiceMap, addr)
    addr == :y ? assignment.y_assignment : throw(KeyError(addr))
end
Gen.get_internal_nodes(assignment::CollapsedHMMChoiceMap) = ((:y, assignment.y_assignment),)

struct CollapsedHMM <: GenerativeFunction{PersistentVector{Float64},CollapsedHMMTrace} end
collapsed_hmm = CollapsedHMM()

function unbiased_logpdf_est(args, ys::PersistentVector{Float64})
    (var_x, var_y, T, num_particles, ess) = args
	local lml_estimate::Float64
	lml_estimate = smc(var_x, var_y, T, num_particles, ess, ys)
    retval = PersistentVector{Float64}(ys)
    vector = VectorDistTrace(retval, args, lml_estimate, length(ys))
    CollapsedHMMTrace(vector)
end

function Gen.generate(generator::CollapsedHMM, args, constraints)
    (var_x, var_y, T, num_particles, ess) = args
    if isempty(constraints)
        error("Unsupported constraints")
    end
    ys = PersistentVector{Float64}(get_leaf_node(constraints, :y => t) for t=1:T)
    if var_x < 0 || var_y < 0
        retval = PersistentVector{Float64}(ys)
        vector = VectorDistTrace(retval, args, -Inf, length(ys))
        trace = CollapsedHMMTrace(vector)
        weight = -Inf
    else
        trace = unbiased_logpdf_est(args, ys)
        weight = get_call_record(trace).score
    end
    (trace, weight)
end

function Gen.update(generator::CollapsedHMM, new_args, args_change, trace, constraints)
    (var_x, var_y, T, num_particles, ess) = new_args
    if !isempty(constraints)
        error("Unsupported constraints")
    end
    ys = trace.vector.values
    if var_x < 0 || var_y < 0
        retval = get_call_record(trace).retval
        T != length(retval) && error("cannot change length")
        vector = VectorDistTrace(retval, new_args, -Inf, length(retval))
        new_trace = CollapsedHMMTrace(vector)
        weight = -Inf
    else
        new_trace = unbiased_logpdf_est(new_args, ys)
        weight = get_call_record(new_trace).score - get_call_record(trace).score
    end
    (new_trace, weight, EmptyChoiceMap(), nothing)
end

@gen (static) function model_collapsed(T::Int)
    var_x::Float64 = @trace(gamma(1, 1), :var_x)
    var_y::Float64 = @trace(gamma(1, 1), :var_y)
    @trace(collapsed_hmm(var_x, var_y, T, 4096, 2048), :hmm)
end
