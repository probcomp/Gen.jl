function importance_sampling(model::GenerativeFunction{T,U}, model_args::Tuple,
                             observations::Assignment,
                             num_samples::Int) where {T,U}
    traces = Vector{U}(undef, num_samples)
    log_weights = Vector{Float64}(undef, num_samples)
    for i=1:num_samples
        (traces[i], log_weights[i]) = initialize(model, model_args, observations)
    end
    log_total_weight = logsumexp(log_weights)
    log_ml_estimate = log_total_weight - log(num_samples)
    log_normalized_weights = log_weights .- log_total_weight
    return (traces, log_normalized_weights, log_ml_estimate)
end

function importance_sampling(model::GenerativeFunction{T,U}, model_args::Tuple,
                             observations::Assignment,
                             proposal::GenerativeFunction, proposal_args::Tuple,
                             num_samples::Int) where {T,U}
    traces = Vector{U}(undef, num_samples)
    log_weights = Vector{Float64}(undef, num_samples)
    for i=1:num_samples
        (proposed_assmt, proposal_weight, _) = propose(proposal, proposal_args)
        constraints = merge(observations, proposed_assmt)
        (traces[i], model_weight) = initialize(model, model_args, constraints)
        log_weights[i] = model_weight - proposal_weight
    end
    log_total_weight = logsumexp(log_weights)
    log_ml_estimate = log_total_weight - log(num_samples)
    log_normalized_weights = log_weights .- log_total_weight
    return (traces, log_normalized_weights, log_ml_estimate)
end

function importance_resampling(model::GenerativeFunction{T,U}, model_args::Tuple,
                               observations::Assignment,
                               num_samples::Int; verbose=false)  where {T,U,V,W}
    (model_trace::U, log_weight) = initialize(model, model_args, observations)
    log_total_weight = log_weight
    for i=2:num_samples
        verbose && println("sample: $i of $num_samples")
        (cand_model_trace, log_weight) = initialize(model, model_args, observations)
        log_total_weight = logsumexp(log_total_weight, log_weight)
        if bernoulli(exp(log_weight - log_total_weight))
            model_trace = cand_model_trace
        end
    end
    log_ml_estimate = log_total_weight - log(num_samples)
    return (model_trace::U, log_ml_estimate::Float64)
end

function importance_resampling(model::GenerativeFunction{T,U}, model_args::Tuple,
                               observations::Assignment,
                               proposal::GenerativeFunction{V,W}, proposal_args::Tuple,
                               num_samples::Int; verbose=false)  where {T,U,V,W}
    (proposal_assmt, proposal_weight, _) = propose(proposal, proposal_args)
    constraints = merge(observations, proposal_assmt)
    (model_trace::U, model_weight) = initialize(model, model_args, constraints)
    log_total_weight = model_weight - proposal_weight
    for i=2:num_samples
        verbose && println("sample: $i of $num_samples")
        (proposal_assmt, proposal_weight, _) = propose(proposal, proposal_args)
        constraints = merge(observations, proposal_assmt)
        (cand_model_trace, model_weight) = initialize(model, model_args, constraints)
        log_weight = model_weight - proposal_weight
        log_total_weight = logsumexp(log_total_weight, log_weight)
        if bernoulli(exp(log_weight - log_total_weight))
            model_trace = cand_model_trace
        end
    end
    log_ml_estimate = log_total_weight - log(num_samples)
    return (model_trace::U, log_ml_estimate::Float64)
end

export importance_sampling, importance_resampling
