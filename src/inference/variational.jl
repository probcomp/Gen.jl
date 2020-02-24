# black box, uses score function estimator
function single_sample_gradient_estimate!(
        var_model::GenerativeFunction, var_model_args::Tuple,
        model::GenerativeFunction, model_args::Tuple, observations::ChoiceMap,
        scale_factor=1.)

    # sample from variational approximation
    trace = simulate(var_model, var_model_args)

    # compute learning signal
    constraints = merge(observations, get_choices(trace)) # TODO characterize what it means when not all var_model choices are in the model..
    (model_log_weight, _) = assess(model, model_args, constraints)
    log_weight = model_log_weight - get_score(trace)

    # accumulate the weighted gradient
    accumulate_param_gradients!(trace, nothing, log_weight * scale_factor)

    # unbiased estimate of objective function, and trace
    (log_weight, trace)
end

function vimco_geometric_baselines(log_weights)
    num_samples = length(log_weights)
    s = sum(log_weights)
    baselines = Vector{Float64}(undef, num_samples)
    for i=1:num_samples
        temp = log_weights[i]
        log_weights[i] = (s - log_weights[i]) / (num_samples - 1)
        baselines[i] = logsumexp(log_weights) - log(num_samples)
        log_weights[i] = temp
    end
    baselines
end

function logdiffexp(x, y)
    m = max(x, y)
    m + log(exp(x - m) - exp(y - m))
end

function vimco_arithmetic_baselines(log_weights)
    num_samples = length(log_weights)
    log_total_weight = logsumexp(log_weights)
    baselines = Vector{Float64}(undef, num_samples)
    for i=1:num_samples
        log_sum_f_without_i = logdiffexp(log_total_weight, log_weights[i])
        log_f_hat = log_sum_f_without_i - log(num_samples - 1)
        baselines[i] = logsumexp(log_sum_f_without_i, log_f_hat) - log(num_samples)
    end
    baselines
end

# black box, VIMCO gradient estimator
# for use in training models
function multi_sample_gradient_estimate!(
        var_model::GenerativeFunction, var_model_args::Tuple,
        model::GenerativeFunction, model_args::Tuple, observations::ChoiceMap,
        num_samples::Int, scale_factor=1., geometric=true)

    # sample from variational approximation multiple times
    traces = Vector{Any}(undef, num_samples)
    log_weights = Vector{Float64}(undef, num_samples)
    for i=1:num_samples
        traces[i] = simulate(var_model, var_model_args)
        constraints = merge(observations, get_choices(traces[i])) # TODO characterize as above
        model_weight, = assess(model, model_args, constraints)
        log_weights[i] = model_weight - get_score(traces[i])
    end

    # multi-sample log marginal likelihood estimate
    log_total_weight = logsumexp(log_weights)
    L = log_total_weight - log(num_samples)

    # baselines
    if geometric
        baselines = vimco_geometric_baselines(log_weights)
    else
        baselines = vimco_arithmetic_baselines(log_weights)
    end

    weights_normalized = exp.(log_weights .- log_total_weight)
    for i=1:num_samples
        learning_signal = (L - baselines[i]) - weights_normalized[i]
        accumulate_param_gradients!(traces[i], nothing, learning_signal * scale_factor)
    end

    # collection of traces and normalized importance weights, and estimate of
    # objective function
    (L, traces, weights_normalized)
end

"""
    black_box_vi!(model::GenerativeFunction, args::Tuple,
                  observations::ChoiceMap,
                  var_model::GenerativeFunction, var_model_args::Tuple,
                  update::ParamUpdate;
                  iters=1000, samples_per_iter=100, verbose=false)

Fit the parameters of a generative function (`var_model`) to the posterior distribution implied by the given model and observations using stochastic gradient methods.
"""
function black_box_vi!(model::GenerativeFunction, model_args::Tuple,
                       observations::ChoiceMap,
                       var_model::GenerativeFunction, var_model_args::Tuple,
                       update::ParamUpdate;
                       iters=1000, samples_per_iter=100, verbose=false)

    traces = Vector{Any}(undef, samples_per_iter)
    obj_ests = Vector{Float64}(undef, iters)
    for iter=1:iters

        # compute gradient estimate and objective function estimate
        obj_est = 0.
        # TODO multithread
        for sample=1:samples_per_iter
            (trace, obj) = single_sample_gradient_estimate!(
                var_model, var_model_args,
                model, model_args, observations, 1/samples_per_iter)
            obj_est += obj
            traces[sample] = trace
        end
        obj_est /= samples_per_iter
        obj_ests[iter] = obj_est

        # print it
        verbose && println("iter $iter; est objective: $obj_est")

        # do an update
        apply!(update)
    end
    
    normalized_weights = fill(1/samples_per_iter, samples_per_iter)
    (obj_est, traces, normalized_weights, obj_ests)
end

"""
    black_box_vimco!(
        model::GenerativeFunction, args::Tuple,
        observations::ChoiceMap,
        var_model::GenerativeFunction, var_model_args::Tuple,
        update::ParamUpdate, num_samples::Int;
        iters=1000, samples_per_iter=100, verbose=false)

Fit the parameters of a generative function (`var_model`) to the posterior distribution implied by the given model and observations using stochastic gradient methods applied to the [Variational Inference with Monte Carlo Objectives](https://arxiv.org/abs/1602.06725) lower bound on the marginal likelihood.
"""
function black_box_vimco!(model::GenerativeFunction, model_args::Tuple,
                       observations::ChoiceMap,
                       var_model::GenerativeFunction, var_model_args::Tuple,
                       update::ParamUpdate, num_samples::Int;
                       iters=1000, samples_per_iter=100, verbose=false,
                       geometric=true)
    
    obj_ests = Vector{Float64}(undef, iters)
    for iter=1:iters

        # compute gradient estimate and objective function estimate
        obj_est = 0.
        for sample=1:samples_per_iter
            (obj, traces, weights) = multi_sample_gradient_estimate!(
                var_model, var_model_args,
                model, model_args, observations, num_samples,
                1/samples_per_iter, geometric)
            obj_est += obj
        end
        obj_est /= samples_per_iter
        obj_ests[iter] = obj_est

        # print it
        verbose && println("iter $iter; est objective: $obj_est")

        # do an update
        apply!(update)
    end

    (obj_est, traces, weights, obj_ests)
end

export black_box_vi!, black_box_vimco!
