"""
    train!(optimizer, gen_fn::GenerativeFunction, data_generator::Function)

Train a generative function.

The function `data_generator` is a function of no arguments that returns a tuple `(inputs, constraints)` where `inputs` is a `Tuple` of inputs (arguments) to `gen_fn`, and `constraints` is an `Assignment`.
Use the algorithm defined by `optimizer` to maximize the expected conditional log probability (density) that `gen_fn` generates the assignment `constraints` given inputs, where the expectation is taken under the output distribution of `data_generator`.
This is equivalent to minimizing the expected KL divergence from the conditional distribution `constraints | inputs` of the data generator to the distribution represented by the generative function, where the expectation is taken under the marginal distribution on `inputs` determined by the data generator.
Mutates the trainable parameters of `gen_fn`.
"""
function train! end

export train!

###############################
# stochastic gradient descent #
###############################

mutable struct SGDOptimizer
    step_size_init::Float64
    step_size_beta::Float64
    num_epoch::Int
    epoch_size::Int
    num_minibatch::Int
    minibatch_size::Int
    minibatch_callback::Function
    epoch_callback::Function
    param_names::Set{Symbol}
    t::Int
end

function SGDOptimizer(param_names, num_epoch::Int, epoch_size::Int,
                      num_minibatch::Int, minibatch_size::Int;
                      step_size_init::Real=1., step_size_beta=1000,
    minibatch_callback=(gen_fn, epoch, minibatch, minibatch_inputs, minibatch_assmts, verbose) -> nothing,
    epoch_callback=(gen_fn, epoch, epoch_inputs, epoch_assmts, verbose) -> nothing)
    SGDOptimizer(step_size_init, step_size_beta, num_epoch, epoch_size,
        num_minibatch, minibatch_size,
        minibatch_callback, epoch_callback, Set{Symbol}(param_names), 1)
end

function update_params!(opt::SGDOptimizer, gen_fn::GenerativeFunction)
    step_size = opt.step_size_init * (opt.step_size_beta + 1) / (opt.step_size_beta + opt.t)
    for param_name in opt.param_names
        value = get_param(gen_fn, param_name)
        grad = get_param_grad(gen_fn, param_name)
        set_param!(gen_fn, param_name, value + grad * step_size)
    end
    opt.t += 1
end

function reset_gradients!(opt::SGDOptimizer, gen_fn::GenerativeFunction)
    for param_name in opt.param_names
        zero_param_grad!(gen_fn, param_name)
    end
end

function train!(opt::SGDOptimizer, gen_fn::GenerativeFunction, data_generator::Function;
                verbose::Bool=false)

    for epoch=1:opt.num_epoch

        # generate data for epoch
        if verbose
            println("generating data for epoch $epoch")
        end
        epoch_inputs = Vector{Tuple}(undef, opt.epoch_size)
        epoch_assmts = Vector{Assignment}(undef, opt.epoch_size)
        for i=1:opt.epoch_size
            (epoch_inputs[i], epoch_assmts[i]) = data_generator()
        end

        # train on epoch data
        if verbose
            println("training for epoch $epoch...")
        end
        for minibatch=1:opt.num_minibatch
            permuted = Random.randperm(opt.epoch_size)
            minibatch_idx = permuted[1:opt.minibatch_size]
            minibatch_inputs = epoch_inputs[minibatch_idx]
            minibatch_assmts = epoch_assmts[minibatch_idx]
            for (inputs, constraints) in zip(minibatch_inputs, minibatch_assmts)
                (trace, _) = initialize(gen_fn, inputs, constraints)
                retval_grad = accepts_output_grad(gen_fn) ? zero(get_retval(trace)) : nothing
                backprop_params(trace, retval_grad)
            end
    
            # do the SGD update
            update_params!(opt, gen_fn)

            # custom user callback (which can use the gradient accumulators)
            opt.minibatch_callback(gen_fn, epoch, minibatch, minibatch_inputs, minibatch_assmts, verbose)

            # reset the gradient accumulators to zero
            reset_gradients!(opt, gen_fn)
        end

        # evaluate score on held out data
        avg_score = 0.
        for i=1:opt.epoch_size
            (inputs, constraints) = data_generator()
            (_, weight) = initialize(gen_fn, inputs, constraints)
            avg_score += weight
        end
        avg_score /= opt.epoch_size

        if verbose
            println("epoch $epoch avg score: $avg_score")
        end

        # custom user callback
        opt.epoch_callback(gen_fn, epoch, epoch_inputs, epoch_assmts, verbose)
    end
end

export SGDOptimizer
