########
# MCMC #
########

function mh(model::Generator, proposal::Generator, proposal_args_rest::Tuple, trace)
    model_args = get_call_record(trace).args
    proposal_args_forward = (trace, proposal_args_rest...,)
    forward_trace = simulate(proposal, proposal_args_forward)
    forward_score = get_call_record(forward_trace).score
    constraints = get_choices(forward_trace)
    (new_trace, weight, discard) = update(
        model, model_args, NoChange(), trace, constraints)
    proposal_args_backward = (new_trace, proposal_args_rest...,)
    backward_trace = assess(proposal, proposal_args_backward, discard)
    backward_score = get_call_record(backward_trace).score
    if log(rand()) < weight - forward_score + backward_score
        # accept
        return new_trace
    else
        # reject
        return trace
    end
end

function mh(model::Generator, selector::SelectionFunction, selector_args::Tuple, trace)
    (selection, _) = select(selector, selector_args, get_choices(trace))
    model_args = get_call_record(trace).args
    (new_trace, weight) = regenerate(model, model_args, NoChange(), trace, selection)
    if log(rand()) < weight
        # accept
        return new_trace
    else
        # reject
        return trace
    end
end

function rjmcmc(model, forward, forward_args_rest, backward, backward_args_rest,
                injective, injective_args, trace, correction)
    model_args = get_call_record(trace).args
    model_score = get_call_record(trace).score
    forward_args = (trace, forward_args_rest...,)
    forward_trace = simulate(forward, forward_args)
    forward_score = get_call_record(forward_trace).score
    input = pair(get_choices(trace), get_choices(forward_trace), :model, :proposal)
    (output, logabsdet) = apply(injective, injective_args, input)
    (model_constraints, backward_constraints) = unpair(output, :model, :proposal)
    new_trace = assess(model, model_args, model_constraints)
    new_model_score = get_call_record(new_trace).score
    backward_args = (new_trace, backward_args_rest...,)
    backward_trace = assess(backward, backward_args, backward_constraints)
    backward_score = get_call_record(backward_trace).score
    alpha = new_model_score - model_score - forward_score + backward_score + logabsdet + correction(new_trace)
    if log(rand()) < alpha
        # accept
        return new_trace
    else
        # reject
        return trace
    end
end

function sample_momenta(mass, n::Int)
    Float64[random(normal, 0, mass) for _=1:n]
end

function assess_momenta(momenta, mass)
    logprob = 0.
    for val in momenta
        logprob += logpdf(normal, val, 0, mass)
    end
    logprob
end

function hmc(model::Generator{T,U}, selection::AddressSet, trace::U;
             mass=0.1, L=10, eps=0.1) where {T,U}
    prev_model_score = get_call_record(trace).score
    model_args = get_call_record(trace).args

    # run leapfrog dynamics
    new_trace = trace
    local prev_momenta_score::Float64
    local momenta::Vector{Float64}
    for step=1:L

        # half step on momenta
        (_, values_trie, gradient_trie) = backprop_trace(model, new_trace, selection, nothing)
        values = to_array(values_trie, Float64)
        gradient = to_array(gradient_trie, Float64)
        if step == 1
            momenta = sample_momenta(mass, length(values))
            prev_momenta_score = assess_momenta(momenta, mass)
        else
            momenta += (eps / 2) * gradient
        end

        # full step on positions
        values_trie = from_array(values_trie, values + eps * momenta)

        # half step on momenta
        (new_trace, _, _) = update(model, model_args, NoChange(), new_trace, values_trie)
        (_, _, gradient_trie) = backprop_trace(model, new_trace, selection, nothing)
        gradient = to_array(gradient_trie, Float64)
        momenta += (eps / 2) * gradient
    end

    # assess new model score (negative potential energy)
    new_model_score = get_call_record(new_trace).score

    # assess new momenta score (negative kinetic energy)
    new_momenta_score = assess_momenta(-momenta, mass)

    # accept or reject
    alpha = new_model_score - prev_model_score + new_momenta_score - prev_momenta_score
    if log(rand()) < alpha
        new_trace
    else
        trace
    end
end

function mala(model::Generator{T,U}, selection::AddressSet, trace::U, tau) where {T,U}
    model_args = get_call_record(trace).args
    std = sqrt(2 * tau)

    # forward proposal
    (_, values_trie, gradient_trie) = backprop_trace(model, trace, selection, nothing)
    values = to_array(values_trie, Float64)
    gradient = to_array(gradient_trie, Float64)
    forward_mu = values + tau * gradient
    forward_score = 0.
    proposed_values = Vector{Float64}(undef, length(values))
    for i=1:length(values)
        proposed_values[i] = random(normal, forward_mu[i], std)
        forward_score += logpdf(normal, proposed_values[i], forward_mu[i], std)
    end

    # evaluate model weight
    constraints = from_array(values_trie, proposed_values)
    (new_trace, weight, discard) = update(
        model, model_args, NoChange(), trace, constraints)

    # backward proposal
    (_, _, backward_gradient_trie) = backprop_trace(model, new_trace, selection, nothing)
    backward_gradient = to_array(backward_gradient_trie, Float64)
    @assert length(backward_gradient) == length(values)
    backward_score = 0.
    backward_mu  = proposed_values + tau * backward_gradient
    for i=1:length(values)
        backward_score += logpdf(normal, values[i], backward_mu[i], std)
    end

    # accept or reject
    alpha = weight - forward_score + backward_score
    if log(rand()) < alpha
        return new_trace
    else
        return trace
    end
end

export mh, rjmcmc, hmc, mala

######################
# particle filtering #
######################

function logsumexp(arr)
    min_arr = maximum(arr)
    min_arr + log(sum(exp.(arr .- min_arr)))
end

function effective_sample_size(log_weights::Vector{Float64})
    # assumes weights are normalized
    log_ess = -logsumexp(2. * log_weights)
    exp(log_ess)
end

"""
the first argument to model should be an integer, starting from 1, that indicates the step
get_observations is a function of the step that returns a choice trie
rejuvenation_move is a function of the step and the previous trace, that returns a new trace
"""
function particle_filter(model::Generator{T,U}, num_steps::Int,
                         num_particles::Int, ess_threshold::Real,
                         get_observations::Function,
                         rejuvenation_move::Function=(t,trace) -> trace) where {T,U}

    log_unnormalized_weights = Vector{Float64}(undef, num_particles)
    log_ml_estimate = 0.
    observations = get_observations(1)
    traces = Vector{T}(undef, num_particles)
    next_traces = Vector{T}(undef, num_particles)
    for i=1:num_particles
        (traces[i], log_unnormalized_weights[i]) = generate(model, (1,), observations)
    end
    for step=2:num_steps

        # rejuvenation moves
        for i=1:num_particles
            traces[i] = rejuvenation_move(step, traces[i])
        end

        # compute new weights
        log_total_weight = logsumexp(log_unnormalized_weights)
        log_normalized_weights = log_unnormalized_weights .- log_total_weight

        # resample
        if effective_sample_size(log_normalized_weights) < ess_threshold
            weights = exp.(log_normalized_weights)
            parents = rand(Distributions.Categorical(weights / sum(weights)), num_particles)
            log_ml_estimate += log_total_weight - log(num_particles)
            log_unnormalized_weights = zeros(num_particles)
        else
            parents = 1:num_particles
        end

        # extend by one time step
        observations = get_observations(step)
        args_change = nothing
        for i=1:num_particles
            parent = parents[i]
            (next_traces[i], weight) = extend(model, (step,), args_change, traces[i], observations)
            log_unnormalized_weights[i] += weight
        end
        tmp = traces
        traces = next_traces
        next_traces = tmp
    end

    # finalize estimate of log marginal likelihood
    log_total_weight = logsumexp(log_unnormalized_weights)
    log_normalized_weights = log_unnormalized_weights .- log_total_weight
    log_ml_estimate += log_total_weight - log(num_particles)
    return (traces, log_normalized_weights, log_ml_estimate)
end

export particle_filter




##############################
# Maximum a posteriori (MAP) #
##############################

"""
Backtracking gradient ascent for MAP inference on selected real-valued choices
"""
function map_optimize(model::Generator, selection::AddressSet,
                      trace; max_step_size=0.1, tau=0.5, min_step_size=1e-16, verbose=false)
    model_args = get_call_record(trace).args
    (_, values, gradient) = backprop_trace(model, trace, selection, nothing)
    values_vec = to_array(values, Float64)
    gradient_vec = to_array(gradient, Float64)
    step_size = max_step_size
    score = get_call_record(trace).score
    while true
        new_values_vec = values_vec + gradient_vec * step_size
        values = from_array(values, new_values_vec)
        # TODO discard and weight are not actually needed, there should be a more specialized variant
        (new_trace, _, discard, _) = update(model, model_args, NoChange(), trace, values)
        new_score = get_call_record(new_trace).score
        change = new_score - score
        if verbose
            println("step_size: $step_size, prev score: $score, new score: $new_score, change: $change")
        end
        if change >= 0.
            # it got better, return it
            return new_trace
        elseif step_size < min_step_size
            # it got worse, but we ran out of attempts
            return trace
        end
        
        # try again with a smaller step size
        step_size = tau * step_size
    end
end

export map_optimize
