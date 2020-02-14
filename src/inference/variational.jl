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

function black_box_vi!(model::GenerativeFunction, model_args::Tuple,
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

export black_box_vi!


## VIMCO (without model update, just training amortized inference var_model..)
## Q: but why would we do this? wouldn't we just want to train on simulated data from the model?
## A: because this way we can train using the actual distribution of the data set
#
#function train_vimco!(data_gen::Function, model::GenerativeFunction,
    #variational_approx::GenerativeFunction;
    #iters=1000, num_samples_per_iter=100)
#
    #for iter=1:iters
        #(model_args, observations) = data_gen()
        #var_model_traces = []
        #for sample=1:num_samples_per_iter
            ## sample from the var_model
            #(var_model_traces[sample], _) = initialize(var_model, var_model_args)
    #
            ## compute constraints on model (we assume all var_model addrs are also in the model)
            #constraints = merge(observations, get_choices(var_model_traces[sample]))
        #
            ## obtain model trace
            #(model_weights[sample], _) = assess(model, model_args, constraints)
        #end
        ## TODO compute weight vector for var_model   
        #for sample=1:num_samples_per_iter
            #backprop_params(var_model_traces[sample], var_model_weights[sample])
        #end
#
        ## TODO PUT UPDATE HERE
    #end
    #
#
#end
#
#
#
#
## different training methods accumulate gradients for the trainable parameters
## of the model and/or variational appropximation.
#
## - the user can write arbitrary code that updates the parameters
#
## - we should separate the objective function (represented by the gradient
## accumulators) from the optimization algorithm (what happens in the update). example:
    #
    ## SGD update
    ## ADAM update
#
    ## (in general, the updates can have state)
#
    ## like TensorFlow's optimizer 'apply_gradients' method, we could provide a
    ## list of the parameters that we want to optimize
#
    ## - this would be a list of generative functions, for each generative
    ## function, a list of its trainable parameters to update
#
    ## but, how do we make it generic for both regular gen functions and TFFunctions?
    ## we can implement sgd_update! and adam_update! methods for each generative function type that takes a list of names of the parameters.
    ## they also need to all have state...
    ## opt = SGDOptimizer(gen_fn, params...); apply_update!(opt)
    ## opt = AdamOptimizer(gen_fn, params...); apply_update!(opt)
    ## for Gen functions, it takes a list of symbols; for the TFFunction, it
    ## will take the PyObjects for the trainable parameters instead (?)
#
    ## Q: can we use a lower-level interface for working with trainable
    ## parameters, instead of having separate apply_update! methods for each
    ## optimizer type?
#
    ## A: one difference is that the Julia operations can be applied one at a
    ## time, whereas the TF operations should be aggregated into one operation
    ## for performance?
#
    ## ex:
    #
    ## - get_param_value(gen_fn, param_name) --> would return Julia reference vs TensrFlow reference
    ## - get_param_grad(gen_fn, param_name) --> ..
    ## - reset_param_grads!(gen_fn, param_name)
#
    ## then, there is also a higher-level constructor for e.g. SGDOptimizer that
    ## takes a list of (gen_fn, params_list) tuples.
#
    ## opt = SGDOptimizer((foo, [a, b, c]), (bar, [d, e, f]))
    ## opt = (foo => [a, b, c], bar => [d, e, f])
    ## train_vae!(opt, data_gen, model, variational_approx)
    ## internally, it will construct an SGDOptimizer for each generative function.
#
    ## or, a lower level interface would just accept a list of optimizers..
    #
    ## opt = 
#
## - 
