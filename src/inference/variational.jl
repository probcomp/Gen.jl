"""
    black_box_vi!(model::GenerativeFunction, args::Tuple,
                  observations::ChoiceMap,
                  proposal::GenerativeFunction, proposal_args::Tuple,
                  update::ParamUpdate;
                  iters=1000, samples_per_iter=100, verbose=false)

Fit the parameters of a generative function (`proposal`) to the posterior distribution implied by the given model and observations using stochastic gradient methods.
"""
function black_box_vi!(model::GenerativeFunction, args::Tuple,
                       observations::ChoiceMap,
                       proposal::GenerativeFunction, proposal_args::Tuple,
                       update::ParamUpdate;
                       iters=1000, samples_per_iter=100, verbose=false)
    for iter=1:iters
        for sample=1:samples_per_iter
            # sample from the proposal
            (trace, _) = generate(proposal, proposal_args)
    
            # compute constraints on model (we assume all proposal addrs are also in the model)
            constraints = merge(observations, get_choices(trace))
    
            # compute log importance weight
            (model_log_weight, _) = assess(model, args, constraints)
            log_weight = model_log_weight - get_score(trace)
    
            # accumulate the weighted gradient
            retval_grad = accepts_output_grad(get_gen_fn(trace)) ? zero(get_retval(trace)) : nothing
            accumulate_param_gradients!(trace, retval_grad, log_weight)
        end

        # do an update
        apply!(update)

        # evaluate score with different samples
        avg_log_weight = 0.
        for i=1:samples_per_iter
            (trace, _) = generate(proposal, proposal_args)
            constraints = merge(observations, get_choices(trace))
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

#########
# VIMCO #
#########

function train_vimco!(data_gen::Function, model::GenerativeFunction,
    variational_approx::GenerativeFunction;
    iters=1000, num_samples_per_iter=100)

    for iter=1:iters
        (model_args, observations) = data_gen()
        proposal_traces = []
        model_traces = []
        for sample=1:num_samples_per_iter
            # sample from the proposal
            (proposal_traces[sample], _) = initialize(proposal, proposal_args)
    
            # compute constraints on model (we assume all proposal addrs are also in the model)
            constraints = merge(observations, get_choices(proposal_traces[sample]))
        
            # obtain model trace
            (model_traces[sample], _) = initialize(model, model_args, constraints)
        end
        # TODO compute weight vector for model and weight vector for proposal   
        for sample=1:num_samples_per_iter
            backprop_params(model_traces[sample], model_weights[sample])
            backprop_params(proposal_traces[sample], proposal_weights[sample])
        end

        # TODO PUT UPDATE HERE
    end
    

end
#
## VIMCO (without model update, just training amortized inference proposal..)
## Q: but why would we do this? wouldn't we just want to train on simulated data from the model?
## A: because this way we can train using the actual distribution of the data set
#
#function train_vimco!(data_gen::Function, model::GenerativeFunction,
    #variational_approx::GenerativeFunction;
    #iters=1000, num_samples_per_iter=100)
#
    #for iter=1:iters
        #(model_args, observations) = data_gen()
        #proposal_traces = []
        #for sample=1:num_samples_per_iter
            ## sample from the proposal
            #(proposal_traces[sample], _) = initialize(proposal, proposal_args)
    #
            ## compute constraints on model (we assume all proposal addrs are also in the model)
            #constraints = merge(observations, get_choices(proposal_traces[sample]))
        #
            ## obtain model trace
            #(model_weights[sample], _) = assess(model, model_args, constraints)
        #end
        ## TODO compute weight vector for proposal   
        #for sample=1:num_samples_per_iter
            #backprop_params(proposal_traces[sample], proposal_weights[sample])
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
