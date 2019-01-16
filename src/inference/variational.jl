"""
    black_box_vi!(model::GenerativeFunction, args::Tuple,
                  observations::Assignment,
                  proposal::GenerativeFunction, proposal_args::Tuple,
                  update::ParamUpdate;
                  iters=1000, samples_per_iter=100, verbose=false)

Fit the parameters of a generative function (`proposal`) to the posterior distribution implied by the given model and observations using stochastic gradient methods.
"""
function black_box_vi!(model::GenerativeFunction, args::Tuple,
                       observations::Assignment,
                       proposal::GenerativeFunction, proposal_args::Tuple,
                       update::ParamUpdate;
                       iters=1000, samples_per_iter=100, verbose=false)
    for iter=1:iters
        for sample=1:samples_per_iter
            # sample from the proposal
            (trace, _) = initialize(proposal, proposal_args)
    
            # compute constraints on model (we assume all proposal addrs are also in the model)
            constraints = merge(observations, get_assmt(trace))
    
            # compute log importance weight
            (model_log_weight, _) = assess(model, args, constraints)
            log_weight = model_log_weight - get_score(trace)
    
            # accumulate the weighted gradient
            retval_grad = accepts_output_grad(get_gen_fn(trace)) ? zero(get_retval(trace)) : nothing
            backprop_params(trace, retval_grad, log_weight)
        end

        # do an update
        apply!(update)

        # evaluate score with different samples
        avg_log_weight = 0.
        for i=1:samples_per_iter
            (trace, _) = initialize(proposal, proposal_args)
            constraints = merge(observations, get_assmt(trace))
            (model_log_weight, _) = assess(model, args, constraints)
            log_weight = model_log_weight - get_score(trace)
            avg_log_weight += log_weight
        end
        avg_log_weight /= samples_per_iter

        if verbose
            println("iter $iter avg log weight: $avg_log_weight")
        end
    end
end

export black_box_vi!
