using Gen
using FunctionalCollections: PersistentVector

####################
# generative model #
####################

x_mean(x_prev::Real, t::Int) = (x_prev / 2.) + 25 * (x_prev / (1 + x_prev * x_prev)) + 8 * cos(1.2 * t)
y_mean(x::Real) = (x * x / 20.)

@gen function hmm(var_x, var_y, T::Int)
    xs = Vector{Float64}(undef, T)
    ys = Vector{Float64}(undef, T)
    xs[1] = @addr(normal(0, 5), :x => 1)
    ys[1] = @addr(normal(y_mean(xs[1]), sqrt(var_y)), :y => 1)
    for t=2:T
        xs[t] = @addr(normal(x_mean(xs[t-1], t), sqrt(var_x)), :x => t)
        ys[t] = @addr(normal(y_mean(xs[t]), sqrt(var_y)), :y => t)
    end
    return (xs, ys)
end

@gen function model(T::Int)
    var_x = @addr(Gen.gamma(1, 1), :var_x)
    var_y = @addr(Gen.gamma(1, 1), :var_y)
    @addr(hmm(var_x, var_y, T), :hmm)
end

####################################
# handcoded sequential Monte Carlo #
####################################

include("smc.jl")

struct SMCParams
    num_particles::Int
    ess::Float64
end

struct HandcodedSMC <: StateSpaceSMCScheme{Float64}
    params::SMCParams
    var_x::Float64
    var_y::Float64
    ys::PersistentVector{Float64}
end

function init(smc::HandcodedSMC)
    state = random(normal, 0, 5)
    log_weight = logpdf(normal, smc.ys[1], y_mean(state), sqrt(smc.var_y))
    (state, log_weight)
end

function init_score(smc::HandcodedSMC, state::Float64)
    logpdf(normal, smc.ys[1], y_mean(state), sqrt(smc.var_y))
end

function forward(smc::HandcodedSMC, prev_state::Float64, t::Int)
	state = random(normal, x_mean(prev_state, t), sqrt(smc.var_x))
    log_weight = logpdf(normal, smc.ys[t], y_mean(state), sqrt(smc.var_y))
    (state, log_weight)
end

function forward_score(smc::HandcodedSMC, prev_state::Float64, state::Float64, t::Int)
	logpdf(normal, state, x_mean(prev_state, t), sqrt(smc.var_x))
end

get_num_steps(smc::HandcodedSMC) = length(smc.ys)
get_num_particles(smc::HandcodedSMC) = smc.params.num_particles
get_ess_threshold(smc::HandcodedSMC) = smc.params.ess


const num_particles = 1000
const ess = 500
const smc_params = SMCParams(num_particles, ess)

##########################################
# collapsed generative model (handcoded) #
##########################################

using Gen: VectorDistTrace, VectorDistTraceChoiceTrie
import Gen: get_call_record, has_choices, get_choices, simulate, assess

struct CollapsedHMMTrace
    vector::VectorDistTrace{Float64}
end

get_call_record(trace::CollapsedHMMTrace) = trace.vector.call
has_choices(trace::CollapsedHMMTrace) = length(trace.vector.call.retval) > 0
get_choices(trace::CollapsedHMMTrace) = CollapsedHMMChoices(get_choices(trace.vector))

struct CollapsedHMMChoices <: ChoiceTrie
    y_choices::VectorDistTraceChoiceTrie
end

# addrs are: :y => i

Base.isempty(trie::CollapsedHMMChoices) = (length(trie.trace.call.retval) > 0)
Gen.get_address_schema(::Type{CollapsedHMMChoices}) = DynamicAddressSchema()
Gen.has_internal_node(trie::CollapsedHMMChoices, addr) = (addr == :y)
function Gen.get_internal_node(trie::CollapsedHMMChoices, addr)
    addr == :y ? trie.y_choices : throw(KeyError(addr))
end
Gen.get_internal_nodes(trie::CollapsedHMMChoices) = ((:y, trie.y_choices),)

struct CollapsedHMM <: Generator{PersistentVector{Float64},CollapsedHMMTrace} end
collapsed_hmm = CollapsedHMM()
Gen.get_concrete_argument_types(CollapsedHMM) = [:Float64, :Float64, :Int, :Int, :Float64]

#function Gen.simulate(generator::CollapsedHMM, args, constraints, read_trace=nothing)
    #(var_x, var_y, T, num_particles, ess) = args
    #hmm_choices = get_choices(simulate(hmm, (var_x, var_y, T)))
    #xs = Float64[hmm_choices[:x => t] for t=1:T]
    #ys = Float64[hmm_choices[:y => t] for t=1:T]
    #smc_scheme = HandcodedSMC(SMCParams(num_particles, ess), var_x, var_y, ys)
    #try
        #smc_result = conditional_smc(smc_scheme, xs)
        #lml_estimate = smc_result.log_ml_estimate
    #catch
        #lml_estimate = -Inf
    #end
    #retval = PersistentVector{Float64}(ys)
    #call = CallRecord(lml_estimate, retval, args)
    #vector = VectorDistTrace(retval, call)
    #CollapsedHMMTrace(vector)
#end

function unbiased_logpdf_est(args, ys::PersistentVector{Float64})
    (var_x, var_y, T, num_particles, ess) = args
	smc_scheme = HandcodedSMC(SMCParams(num_particles, ess), var_x, var_y, ys)
	local lml_estimate::Float64
	try
		smc_result = smc(smc_scheme)
		lml_estimate = smc_result.log_ml_estimate
	catch
		lml_estimate = -Inf
	end
    retval = PersistentVector{Float64}(ys)
    call = CallRecord(lml_estimate, retval, args)
    vector = VectorDistTrace(retval, call)
    CollapsedHMMTrace(vector)
end 

function Gen.generate(generator::CollapsedHMM, args, constraints, read_trace=nothing)
    (var_x, var_y, T, num_particles, ess) =args 
    if isempty(constraints)
        error("Unsupported constraints")
    end
    ys = PersistentVector{Float64}(get_leaf_node(constraints, :y => t) for t=1:T)
    if var_x < 0 || var_y < 0
        retval = PersistentVector{Float64}(ys)
        call = CallRecord(-Inf, retval, args)
        vector = VectorDistTrace(retval, call)
        trace = CollapsedHMMTrace(vector)
        weight = -Inf
    else
        trace = unbiased_logpdf_est(args, ys)
        weight = get_call_record(trace).score
    end
    (trace, weight)
end

function Gen.update(generator::CollapsedHMM, new_args, args_change, trace, constraints, args)
    (var_x, var_y, T, num_particles, ess) = new_args
    if !isempty(constraints)
        error("Unsupported constraints")
    end
    ys = trace.vector.values
    if var_x < 0 || var_y < 0
        retval = get_call_record(trace).retval
        call = CallRecord(-Inf, retval, new_args)
        vector = VectorDistTrace(retval, call)
        new_trace = CollapsedHMMTrace(vector)
        weight = -Inf
    else
        new_trace = unbiased_logpdf_est(new_args, ys)
        weight = get_call_record(new_trace).score - get_call_record(trace).score
    end
    (new_trace, weight, EmptyChoiceTrie(), nothing)
end

@gen function model_collapsed(T::Int)
    var_x = @addr(Gen.gamma(1, 1), :var_x)
    var_y = @addr(Gen.gamma(1, 1), :var_y)
    @addr(collapsed_hmm(var_x, var_y, T, num_particles, ess), :hmm)
end
