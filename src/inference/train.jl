"""
    train!(gen_fn::GenerativeFunction, data_generator::Function,
           update::ParamUpdate,
           num_epoch, epoch_size, num_minibatch, minibatch_size; verbose::Bool=false)

Train the given generative function to maximize the expected conditional log probability (density) that `gen_fn` generates the assignment `constraints` given inputs, where the expectation is taken under the output distribution of `data_generator`.

The function `data_generator` is a function of no arguments that returns a tuple `(inputs, constraints)` where `inputs` is a `Tuple` of inputs (arguments) to `gen_fn`, and `constraints` is an `ChoiceMap`.
`conf` configures the optimization algorithm used.
`param_lists` is a map from generative function to lists of its parameters.
This is equivalent to minimizing the expected KL divergence from the conditional distribution `constraints | inputs` of the data generator to the distribution represented by the generative function, where the expectation is taken under the marginal distribution on `inputs` determined by the data generator.
"""
function train!(gen_fn::GenerativeFunction, data_generator::Function,
                update::ParamUpdate;
                num_epoch=1, epoch_size=1, num_minibatch=1, minibatch_size=1,
                evaluation_size=epoch_size, verbose=false)

    history = Vector{Float64}(undef, num_epoch)
    for epoch=1:num_epoch

        # generate data for epoch
        if verbose
            println("epoch $epoch: generating $epoch_size training examples...")
        end
        epoch_inputs = Vector{Tuple}(undef, epoch_size)
        epoch_choice_maps = Vector{ChoiceMap}(undef, epoch_size)
        for i=1:epoch_size
            (epoch_inputs[i], epoch_choice_maps[i]) = data_generator()
        end

        # train on epoch data
        if verbose
            println("epoch $epoch: training using $num_minibatch minibatches of size $minibatch_size...")
        end
        for minibatch=1:num_minibatch
            permuted = Random.randperm(epoch_size)
            minibatch_idx = permuted[1:minibatch_size]
            minibatch_inputs = epoch_inputs[minibatch_idx]
            minibatch_choice_maps = epoch_choice_maps[minibatch_idx]
            for (inputs, constraints) in zip(minibatch_inputs, minibatch_choice_maps)
                (trace, _) = generate(gen_fn, inputs, constraints)
                accumulate_param_gradients!(trace)
            end
            apply!(update)
        end

        # evaluate score on held out data
        if verbose
            println("epoch $epoch: evaluating on $evaluation_size examples...")
        end
        avg_score = 0.
        for i=1:evaluation_size
            (inputs, constraints) = data_generator()
            (_, weight) = generate(gen_fn, inputs, constraints)
            avg_score += weight
        end
        avg_score /= evaluation_size
        
        history[epoch] = avg_score

        if verbose
            println("epoch $epoch: est. objective value: $avg_score")
        end
    end
    return history
end


"""
    score = lecture!(
        p::GenerativeFunction, p_args::Tuple,
        q::GenerativeFunction, get_q_args::Function)

Simulate a trace of p representing a training example, and use to update the gradients of the trainable parameters of q.

Used for training q via maximum expected conditional likelihood.
Random choices will be mapped from p to q based on their address.
get_q_args maps a trace of p to an argument tuple of q.
score is the conditional log likelihood (or an unbiased estimate of a lower bound on it, if not all of q's random choices are constrained, or if q uses non-addressable randomness).
"""
function lecture!(
        p::GenerativeFunction, p_args::Tuple,
        q::GenerativeFunction, get_q_args::Function)
    p_trace = simulate(p, p_args)
    q_args = get_q_args(p_trace)
    q_trace, score = generate(q, q_args, get_choices(p_trace)) # NOTE: q won't make all the random choices that p does
    accumulate_param_gradients!(q_trace)
    score
end

"""
    score = lecture_batched!(
        p::GenerativeFunction, p_args::Tuple,
        q::GenerativeFunction, get_q_args::Function)

Simulate a batch of traces of p representing training samples, and use them to update the gradients of the trainable parameters of q.

Like `lecture!` but q is batched, and must make random choices for training sample i under hierarchical address namespace i::Int (e.g. i => :z).
get_q_args maps a vector of traces of p to an argument tuple of q.
"""
function lecture_batched!(
        p::GenerativeFunction, p_args::Tuple,
        q_batched::GenerativeFunction, get_q_args::Function,
        batch_size::Int)
    p_traces = Vector{Any}(undef, batch_size)
    constraints = choicemap()
    for i=1:batch_size
        p_trace = simulate(p, p_args)
        set_submap!(constraints, i, get_choices(p_trace))
        p_traces[i] = p_trace
    end
    q_args = get_q_args(p_traces)
    q_trace, score = generate(q_batched, q_args, constraints) # NOTE: q won't make all the random choices that p does
    accumulate_param_gradients!(q_trace)
    score / batch_size
end

# instead of this, write a page documenting the design patterns for learning..

# note: this can also be used for amortized VI i.e. VAEs:
# in the non-amortized case, get_data just returns the same model_args
# (so there is one variational approximation we are fitting to all the posteriors)

# TODO we need get_model_args if we are training conditional model
# TODO abstract out the VI piece?
function learn_from_incomplete_data!(
        model::GenerativeFunction, get_data::Function,
        update_model!::Function, # provide a default
        stop::Function=(iter, obj_estimates) -> iter > 100)
        #num_epoch=1, epoch_size=1, num_minibatch=1, minibatch_size=1,
        #evaluation_size=epoch_size, verbose=false)

    # non-amortized version
    # if there are lots of data points, they all are in the same choice map
    #(model_args, observations, var_model_args) = get_data()
    iter = 1
    obj_estimates = Float64[]
    while !stop(iter, obj_estimates)
        # amortized version
        #(model_args, observations, var_model_args) = get_data()

        (inference_traces, inference_weights) = do_inference(
            

        # train the variational approximation to the current posterior
        # TODO support amortized variational approximation i.e. VAE
        # TODO is any inference procedure that returns a weighted collection of traces valid here?
        (elbo_est, var_traces, weights, _) = black_box_vi!(
            model, model_args, observations,
            var_model, var_model_args, var_update) # num_samples, iters, samples_per_iter, ..

        # NOTE: the state of BBVI stays around, which is what we want.

        # update the model using a stochastic estimate of the gradient of a
        # lower bound on the marginal likelhood, using inferred traces
        for (inference_trace, weight) in zip(inference_traces, weights)
            constraints = merge(observations, get_choices(var_trace))
            (model_trace, _) = generate(model, model_args, constraints)
            accumulate_param_gradients!(model_trace, nothing, weight)
        end
        update_model!()
    end
    iter += 1
end



export train!
export lecture!
export lecture_batched!
