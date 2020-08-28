import Distributions

function effective_sample_size(log_normalized_weights::Vector{Float64})
    log_ess = -logsumexp(2. * log_normalized_weights)
    return exp(log_ess)
end

function normalize_weights(log_weights::Vector{Float64})
    log_total_weight = logsumexp(log_weights)
    log_normalized_weights = log_weights .- log_total_weight
    return (log_total_weight, log_normalized_weights)
end

#######################################
# building blocks for particle filter #
#######################################

mutable struct ParticleFilterState{U}
    traces::Vector{U}
    new_traces::Vector{U}
    log_weights::Vector{Float64}
    log_ml_est::Float64
    parents::Vector{Int}
end

"""
    traces = get_traces(state::ParticleFilterState)

Return the vector of traces in the current state, one for each particle.
"""
function get_traces(state::ParticleFilterState)
    state.traces

end

"""
    log_weights = get_log_weights(state::ParticleFilterState)

Return the vector of log weights for the current state, one for each particle.

The weights are not normalized, and are in log-space.
"""
function get_log_weights(state::ParticleFilterState)
    state.log_weights
end

"""
    estimate = log_ml_estimate(state::ParticleFilterState)

Return the particle filter's current estimate of the log marginal likelihood.
"""
function log_ml_estimate(state::ParticleFilterState)
    num_particles = length(state.traces)
    return state.log_ml_est + logsumexp(state.log_weights) - log(num_particles)
end

"""
    traces::Vector = sample_unweighted_traces(state::ParticleFilterState, num_samples::Int)

Sample a vector of `num_samples` traces from the weighted collection of traces in the given particle filter state.
"""
function sample_unweighted_traces(state::ParticleFilterState{U}, num_samples::Int) where {U}
    (_, log_normalized_weights) = normalize_weights(state.log_weights)
    weights = exp.(log_normalized_weights)
    traces = Vector{U}(undef, num_samples)
    for i=1:num_samples
        traces[i] = state.traces[categorical(weights)]
    end
    return traces
end

"""
    state = initialize_particle_filter(model::GenerativeFunction, model_args::Tuple,
        observations::ChoiceMap proposal::GenerativeFunction, proposal_args::Tuple,
        num_particles::Int)

Initialize the state of a particle filter using a custom proposal for the initial latent state.
"""
function initialize_particle_filter(model::GenerativeFunction{T,U}, model_args::Tuple,
        observations::ChoiceMap, proposal::GenerativeFunction, proposal_args::Tuple,
        num_particles::Int) where {T,U}
    traces = Vector{Any}(undef, num_particles)
    log_weights = Vector{Float64}(undef, num_particles)
    for i=1:num_particles
        (prop_choices, prop_weight, _) = propose(proposal, proposal_args)
        (traces[i], model_weight) = generate(model, model_args, merge(observations, prop_choices))
        log_weights[i] = model_weight - prop_weight
    end
    ParticleFilterState{U}(traces, Vector{U}(undef, num_particles),
        log_weights, 0., collect(1:num_particles))
end

"""
    state = initialize_particle_filter(model::GenerativeFunction, model_args::Tuple,
        observations::ChoiceMap, num_particles::Int)

Initialize the state of a particle filter, using the default proposal for the initial latent state.
"""
function initialize_particle_filter(model::GenerativeFunction{T,U}, model_args::Tuple,
        observations::ChoiceMap, num_particles::Int) where {T,U}
    traces = Vector{Any}(undef, num_particles)
    log_weights = Vector{Float64}(undef, num_particles)
    for i=1:num_particles
        (traces[i], log_weights[i]) = generate(model, model_args, observations)
    end
    ParticleFilterState{U}(traces, Vector{U}(undef, num_particles),
        log_weights, 0., collect(1:num_particles))
end

"""
    particle_filter_step!(state::ParticleFilterState, new_args::Tuple, argdiffs,
        observations::ChoiceMap, proposal::GenerativeFunction, proposal_args::Tuple)

Perform a particle filter update, where the model arguments are adjusted, new
observations are added, and some combination of a custom proposal and the
model's internal proposal is used for proposing new latent state.  That is, for
each particle,

* The proposal function `proposal` is evaluated with arguments `Tuple(t_old,
  proposal_args...)` (where `t_old` is the old model trace), and produces its
  own trace (call it `proposal_trace`); and
* The old model trace is replaced by a new model trace (call it `t_new`).

The choicemap of `t_new` satisfies the following conditions:

1. `get_choices(t_old)` is a subset of `get_choices(t_new)`;
2. `observations` is a subset of `get_choices(t_new)`;
3. `get_choices(proposal_trace)` is a subset of `get_choices(t_new)`.

Here, when we say one choicemap `a` is a "subset" of another choicemap `b`, we
mean that all keys that occur in `a` also occur in `b`, and the values at those
addresses are equal.

It is an error if no trace `t_new` satisfying the above conditions exists in
the support of the model (with the new arguments). If such a trace exists, then
the random choices not determined by the above requirements are sampled using
the internal proposal.
"""
function particle_filter_step!(state::ParticleFilterState{U}, new_args::Tuple, argdiffs::Tuple,
        observations::ChoiceMap, proposal::GenerativeFunction, proposal_args::Tuple) where {U}
    num_particles = length(state.traces)
    for i=1:num_particles
        (prop_choices, prop_weight, _) = propose(proposal, (state.traces[i], proposal_args...))
        constraints = merge(observations, prop_choices)
        (state.new_traces[i], up_weight, _, disc) = update(state.traces[i], new_args, argdiffs, constraints)
        @assert isempty(disc)
        state.log_weights[i] += up_weight - prop_weight
    end

    # swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp

    return nothing
end

"""
    particle_filter_step!(state::ParticleFilterState, new_args::Tuple, argdiffs,
        observations::ChoiceMap)

Perform a particle filter update, where the model arguments are adjusted, new observations are added, and the default proposal is used for new latent state.
"""
function particle_filter_step!(state::ParticleFilterState{U}, new_args::Tuple, argdiffs::Tuple,
        observations::ChoiceMap) where {U}
    num_particles = length(state.traces)
    for i=1:num_particles
        (state.new_traces[i], increment, _, discard) = update(
            state.traces[i], new_args, argdiffs, observations)
        if !isempty(discard)
            error("Choices were updated or deleted inside particle filter step: $discard")
        end
        state.log_weights[i] += increment
    end

    # swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp

    return nothing
end

"""
    did_resample::Bool = maybe_resample!(state::ParticleFilterState;
        ess_threshold::Float64=length(state.traces)/2, verbose=false)

Do a resampling step if the effective sample size is below the given threshold.
Return `true` if a resample thus occurred, `false` otherwise.
"""
function maybe_resample!(state::ParticleFilterState{U};
                        ess_threshold::Real=length(state.traces)/2, verbose=false) where {U}
    num_particles = length(state.traces)
    (log_total_weight, log_normalized_weights) = normalize_weights(state.log_weights)
    ess = effective_sample_size(log_normalized_weights)
    do_resample = ess < ess_threshold
    if verbose
        println("effective sample size: $ess, doing resample: $do_resample")
    end
    if do_resample
        weights = exp.(log_normalized_weights)
        Distributions.rand!(Distributions.Categorical(weights / sum(weights)), state.parents)
        state.log_ml_est += log_total_weight - log(num_particles)
        for i=1:num_particles
            state.new_traces[i] = state.traces[state.parents[i]]
            state.log_weights[i] = 0.
        end

        # swap references
        tmp = state.traces
        state.traces = state.new_traces
        state.new_traces = tmp
    end
    return do_resample
end

export initialize_particle_filter, particle_filter_step!, maybe_resample!
export get_traces, get_log_weights, log_ml_estimate, sample_unweighted_traces
