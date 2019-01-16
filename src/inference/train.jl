"""
    train!(gen_fn::GenerativeFunction, data_generator::Function,
           update::ParamUpdate,
           num_epoch, epoch_size, num_minibatch, minibatch_size; verbose::Bool=false)

Train the given generative function to maximize the expected conditional log probability (density) that `gen_fn` generates the assignment `constraints` given inputs, where the expectation is taken under the output distribution of `data_generator`.

The function `data_generator` is a function of no arguments that returns a tuple `(inputs, constraints)` where `inputs` is a `Tuple` of inputs (arguments) to `gen_fn`, and `constraints` is an `Assignment`.
`conf` configures the optimization algorithm used.
`param_lists` is a map from generative function to lists of its parameters.
This is equivalent to minimizing the expected KL divergence from the conditional distribution `constraints | inputs` of the data generator to the distribution represented by the generative function, where the expectation is taken under the marginal distribution on `inputs` determined by the data generator.
"""
function train!(gen_fn::GenerativeFunction, data_generator::Function,
                update::ParamUpdate,
                num_epoch, epoch_size, num_minibatch, minibatch_size;
                verbose::Bool=false)

    for epoch=1:num_epoch

        # generate data for epoch
        if verbose
            println("generating data for epoch $epoch")
        end
        epoch_inputs = Vector{Tuple}(undef, epoch_size)
        epoch_assmts = Vector{Assignment}(undef, epoch_size)
        for i=1:epoch_size
            (epoch_inputs[i], epoch_assmts[i]) = data_generator()
        end

        # train on epoch data
        if verbose
            println("training for epoch $epoch...")
        end
        for minibatch=1:num_minibatch
            permuted = Random.randperm(epoch_size)
            minibatch_idx = permuted[1:minibatch_size]
            minibatch_inputs = epoch_inputs[minibatch_idx]
            minibatch_assmts = epoch_assmts[minibatch_idx]
            for (inputs, constraints) in zip(minibatch_inputs, minibatch_assmts)
                (trace, _) = initialize(gen_fn, inputs, constraints)
                retval_grad = accepts_output_grad(gen_fn) ? zero(get_retval(trace)) : nothing
                backprop_params(trace, retval_grad)
            end
            apply!(update)
        end

        # evaluate score on held out data
        avg_score = 0.
        for i=1:epoch_size
            (inputs, constraints) = data_generator()
            (_, weight) = initialize(gen_fn, inputs, constraints)
            avg_score += weight
        end
        avg_score /= epoch_size

        if verbose
            println("epoch $epoch avg score: $avg_score")
        end
    end
end

export train!
