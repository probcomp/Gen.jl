function effective_sample_size(log_normalized_weights::Vector{Float64})
    log_ess = -logsumexp(2. * log_normalized_weights)
    exp(log_ess)
end

function normalize_weights(log_unnormalized_weights::Vector{Float64})
    log_total_weight = logsumexp(log_unnormalized_weights)
    log_normalized_weights = log_unnormalized_weights .- log_total_weight
    (log_total_weight, log_normalized_weights)
end

function fill_parents_self!(parents::Vector{Int})
    for i=1:length(parents)
        parents[i] = i
    end
end

"""
the first argument to model should be an integer, starting from 1, that indicates the step
get_observations is a function of the step that returns a choice trie
"""
function particle_filter_default(model::GenerativeFunction{T,U},
                                 model_args_rest::Tuple, num_steps::Int,
                                 num_particles::Int, ess_threshold::Real,
                                 init_observations::Assignment,
                                 step_observations::Function;
                                 verbose::Bool=false) where {T,U}

    log_unnormalized_weights = Vector{Float64}(undef, num_particles)
    log_ml_estimate = 0.
    traces = Vector{U}(undef, num_particles)
    next_traces = Vector{U}(undef, num_particles)
    for i=1:num_particles
        (traces[i], log_unnormalized_weights[i]) = initialize(
            model, (1, model_args_rest...), init_observations)
    end

    parents = Vector{Int}(undef, num_particles)
    for step=2:num_steps

        # compute new weights
        (log_total_weight, log_normalized_weights) = normalize_weights(log_unnormalized_weights)
        ess = effective_sample_size(log_normalized_weights)
        verbose && println("step: $step, ess: $ess")

        # resample
        if ess < ess_threshold
            verbose && println("resampling..")
            weights = exp.(log_normalized_weights)
            Distributions.rand!(Distributions.Categorical(weights / sum(weights)), parents)
            log_ml_estimate += log_total_weight - log(num_particles)
            fill!(log_unnormalized_weights, 0.)
        else
            fill_parents_self!(parents)
        end

        # extend by one time step
        (observations, argdiff) = step_observations(step)
        for i=1:num_particles
            parent = parents[i]
            parent_trace = traces[parent]
            (next_traces[i], log_weight) = extend((step, model_args_rest...),
                                                  argdiff, parent_trace, observations)
            log_unnormalized_weights[i] += log_weight
        end
        tmp = traces
        traces = next_traces
        next_traces = tmp
    end

    # finalize estimate of log marginal likelihood
    (log_total_weight, log_normalized_weights) = normalize_weights(log_unnormalized_weights)
    ess = effective_sample_size(log_normalized_weights)
    log_ml_estimate += log_total_weight - log(num_particles)
    verbose && println("final ess: $ess, final log_ml_est: $log_ml_estimate")
    return (traces, log_normalized_weights, log_ml_estimate)
end

function particle_filter_custom(model::GenerativeFunction{T,U}, model_args_rest::Tuple,
                                num_steps::Int, num_particles::Int, ess_threshold::Real,
                                init_observations::Assignment, init_proposal_args::Tuple,
                                step_observations::Function, step_proposal_args::Function,
                                init_proposal::GenerativeFunction, step_proposal::GenerativeFunction;
                                verbose::Bool=false) where {T,U}

    log_unnormalized_weights = Vector{Float64}(undef, num_particles)
    log_ml_estimate = 0.
    traces = Vector{U}(undef, num_particles)
    next_traces = Vector{U}(undef, num_particles)
    for i=1:num_particles
        (proposal_assmt, proposal_weight) = propose(init_proposal, init_proposal_args)
        constraints = merge(init_observations, proposal_assmt)
        (traces[i], model_weight) = initialize(model, (1, model_args_rest...), constraints)
        log_unnormalized_weights[i] = model_weight - proposal_weight
    end

    parents = Vector{Int}(undef, num_particles)
    for step=2:num_steps

        # compute new weights
        (log_total_weight, log_normalized_weights) = normalize_weights(log_unnormalized_weights)
        ess = effective_sample_size(log_normalized_weights)
        verbose && println("step: $step, ess: $ess")

        # resample
        if ess < ess_threshold
            verbose && println("resampling..")
            weights = exp.(log_normalized_weights)
            Distributions.rand!(Distributions.Categorical(weights / sum(weights)), parents)
            log_ml_estimate += log_total_weight - log(num_particles)
            fill!(log_unnormalized_weights, 0.)
        else
            fill_parents_self!(parents)
        end

        (observations, argdiff) = step_observations(step)

        # extend by one time step
        for i=1:num_particles
            parent = parents[i]
            parent_trace = traces[parent]
            proposal_args = step_proposal_args(step, parent_trace)
            (proposal_assmt, proposal_weight) = propose(step_proposal, proposal_args)
            constraints = merge(observations, proposal_assmt)
            (next_traces[i], model_weight) = extend(
                (step, model_args_rest...), argdiff, parent_trace, constraints)
            log_weight = model_weight - proposal_weight
            log_unnormalized_weights[i] += log_weight
        end
        tmp = traces
        traces = next_traces
        next_traces = tmp
    end

    # finalize estimate of log marginal likelihood
    (log_total_weight, log_normalized_weights) = normalize_weights(log_unnormalized_weights)
    ess = effective_sample_size(log_normalized_weights)
    log_ml_estimate += log_total_weight - log(num_particles)
    verbose && println("final ess: $ess, final log_ml_est: $log_ml_estimate")
    return (traces, log_normalized_weights, log_ml_estimate)
end

export particle_filter_custom
export particle_filter_default
