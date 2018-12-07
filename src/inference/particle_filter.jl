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
rejuvenation_move is a function of the step and the previous trace, that returns a new trace
"""
function particle_filter(model::GenerativeFunction{T,U}, model_args_rest::Tuple, num_steps::Int,
                         num_particles::Int, ess_threshold::Real,
                         get_observations::Function,
                         rejuvenation_move::Function=(t,trace) -> trace;
                         verbose::Bool=false) where {T,U}

    log_unnormalized_weights = Vector{Float64}(undef, num_particles)
    log_ml_estimate = 0.
    (observations, _) = get_observations(1)
    traces = Vector{U}(undef, num_particles)
    next_traces = Vector{U}(undef, num_particles)
    for i=1:num_particles
        (traces[i], log_unnormalized_weights[i]) = initialize(
            model, (1, model_args_rest...), observations)
    end

    parents = Vector{Int}(undef, num_particles)
    for step=2:num_steps

        # rejuvenation moves
        for i=1:num_particles
            traces[i] = rejuvenation_move(step, traces[i])
        end

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
        (observations, argdiff) = get_observations(step)
        for i=1:num_particles
            parent = parents[i]
            parent_trace = traces[parent]
            (next_traces[i], log_weight) = extend(model, (step, model_args_rest...),
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

function particle_filter(model::GenerativeFunction{T,U}, model_args_rest::Tuple,
                         num_steps::Int, num_particles::Int, ess_threshold::Real,
                         get_init_observations_and_proposal_args::Function,
                         get_step_observations_and_proposal_args::Function,
                         init_proposal::GenerativeFunction, step_proposal::GenerativeFunction;
                         verbose::Bool=false) where {T,U}

    log_unnormalized_weights = Vector{Float64}(undef, num_particles)
    log_ml_estimate = 0.
    (observations, proposal_args) = get_init_observations_and_proposal_args()
    traces = Vector{U}(undef, num_particles)
    next_traces = Vector{U}(undef, num_particles)
    for i=1:num_particles
        proposal_trace = simulate(init_proposal, proposal_args)
        proposal_score = get_call_record(proposal_trace).score
        constraints = merge(observations, get_assignment(proposal_trace))
        (traces[i], model_weight) = initialize(model, (1, model_args_rest...), constraints)
        log_unnormalized_weights[i] = model_weight - proposal_score
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
        for i=1:num_particles
            parent = parents[i]
            parent_trace = traces[parent]
            (observations, proposal_args, argdiff) = get_step_observations_and_proposal_args(step, parent_trace)
            proposal_trace = simulate(step_proposal, proposal_args)
            proposal_score = get_call_record(proposal_trace).score
            constraints = merge(observations, get_assignment(proposal_trace))
            (next_traces[i], model_weight) = extend(
                model, (step, model_args_rest...), argdiff, parent_trace, constraints)
            log_weight = model_weight - proposal_score
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

export particle_filter
