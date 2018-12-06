function importance_sampling(model::GenerativeFunction{T,U}, model_args::Tuple,
                             observations::Assignment,
                             num_samples::Int) where {T,U}
    traces = Vector{U}(undef, num_samples)
    log_weights = Vector{Float64}(undef, num_samples)
    for i=1:num_samples
        (traces[i], log_weights[i]) = generate(model, model_args, observations)
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
        proposal_trace = simulate(proposal, proposal_args)
        proposal_score = get_call_record(proposal_trace).score
        constraints = merge(observations, get_assignment(proposal_trace))
        traces[i] = assess(model, model_args, constraints)
        model_score = get_call_record(traces[i]).score
        log_weights[i] = model_score - proposal_score
    end
    log_total_weight = logsumexp(log_weights)
    log_ml_estimate = log_total_weight - log(num_samples)
    log_normalized_weights = log_weights .- log_total_weight
    return (traces, log_normalized_weights, log_ml_estimate)
end

function importance_resampling(model::GenerativeFunction{T,U}, model_args::Tuple,
                               observations::Assignment,
                               num_samples::Int; verbose=false)  where {T,U,V,W}
    (model_trace::U, log_weight) = generate(model, model_args, observations)
    log_total_weight = log_weight
    for i=2:num_samples
        verbose && println("sample: $i of $num_samples")
        (cand_model_trace, log_weight) = generate(model, model_args, observations)
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
    proposal_trace::W = simulate(proposal, proposal_args)
    proposal_score = get_call_record(proposal_trace).score
    constraints::Assignment = merge(observations, get_assignment(proposal_trace))
    (model_trace::U, model_score) = generate(model, model_args, constraints)
    log_total_weight = model_score - proposal_score
    for i=2:num_samples
        verbose && println("sample: $i of $num_samples")
        proposal_trace = simulate(proposal, proposal_args)
        proposal_score = get_call_record(proposal_trace).score
        constraints = merge(observations, get_assignment(proposal_trace))
        (cand_model_trace, model_score) = generate(model, model_args, constraints)
        log_weight = model_score - proposal_score
        log_total_weight = logsumexp(log_total_weight, log_weight)
        if bernoulli(exp(log_weight - log_total_weight))
            model_trace = cand_model_trace
        end
    end
    log_ml_estimate = log_total_weight - log(num_samples)
    return (model_trace::U, log_ml_estimate::Float64)
end

export importance_sampling, importance_resampling
