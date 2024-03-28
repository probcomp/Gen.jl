# Learning Generative Functions

Learning and inference are closely related concepts, and the distinction between the two is not always clear.
Often, **learning** refers to inferring long-lived unobserved quantities that will be reused across many problem instances (like a dynamics model for an entity that we are trying to track), whereas **inference** refers to inferring shorter-lived quantities (like a specific trajectory of a specific entity).
Learning is the way to use data to automatically generate **models** of the world, or to automatically fill in unknown parameters in hand-coded models.
These resulting models are then used in various inference tasks.

There are many variants of the learning task--we could be training the weights of a neural network, estimating a handful of parameters in a structured and hand-coded model, or we could be learning the structure or architecture of a model.
Also, we could do Bayesian learning in which we seek a probability distribution on possible models, or we could seek just the best model, as measured by e.g. **maximum likelihood**.
This section focuses on maximum likelihood learning of the [Trainable parameters](@ref) of a generative function.
These are numerical quantities that are part of the generative function's state, with respect to which generative functions are able to report gradients of their (log) probability density function of their density function.
Trainable parameters are different from random choices--random choices are per-trace and trainable parameters are a property of the generative function (which is associated with many traces).
Also, unlike random choices, trainable parameters do not have a prior distribution.

There are two settings in which we might learn these parameters using maximum likelihood.
If our observed data contains values for all of the random choices made by the generative function, this is called **learning from complete data**, and is a relatively straightforward task.
If our observed data is missing values for some random choices (either because the value happened to be missing, or because it was too expensive to acquire it, or because it is an inherently unmeasurable quantity), this is called **learning from incomplete data**, and is a substantially harder task.
Gen provides programming primitives and design patterns for both tasks.
In both cases, the models we are learning can be either generative or discriminative.

## Learning from Complete Data

This section discusses maximizing the log likelihood of observed data over the space of trainable parameters, when all of the random variables are observed.
In Gen, the likelihood of complete data is simply the joint probability (density) of a trace, and maximum likelihood with complete data amounts to maximizing the sum of log joint probabilities of a collection of traces ``t_i`` for ``i = 1\ldots N`` with respect to the trainable parameters of the generative function, which are denoted ``\theta``.
```math
\max_{\theta} \sum_{i=1}^N \log p(t_i; x, \theta)
```
For example, here is a simple generative model that we might want to learn:
```julia
@gen function model()
    @param x_mu::Float64
    @param a::Float64
    @param b::Float64
    x = @trace(normal(x_mu, 1.), :x)
    @trace(normal(a * x + b, 1.), :y)
end
```
There are three components to ``\theta`` for this generative function: `(x_mu, a, b)`.

Note that maximum likelihood can be used to learn generative and discriminative models, but for discriminative models, the arguments to the generative function will be different for each training example:
```math
\max_{\theta} \sum_{i=1}^N \log p(t_i; x_i, \theta)
```
Here is a minimal discriminative model:
```julia
@gen function disc_model(x::Float64)
    @param a::Float64
    @param b::Float64
    @trace(normal(a * x + b, 1.), :y)
end
```

Let's suppose we are training the generative model.
The first step is to initialize the values of the trainable parameters, which for generative functions constructed using the built-in modeling languages, we do with [`init_param!`](@ref):
```julia
init_param!(model, :a, 0.)
init_param!(model, :b, 0.)
```
Each trace in the collection contains the observed data from an independent draw from our model.
We can populate each trace with its observed data using [`generate`](@ref):
```julia
traces = []
for observations in data
    trace, = generate(model, model_args, observations)
    push!(traces, trace)
end
```
For the complete data case, we assume that all random choices in the model are constrained by the observations choice map (we will analyze the case when not all random choices are constrained in the next section).
We can evaluate the objective function by summing the result of [`get_score`](@ref) over our collection of traces:
```julia
objective = sum([get_score(trace) for trace in traces])
```
We can compute the gradient of this objective function with respect to the trainable parameters using [`accumulate_param_gradients!`](@ref):
```julia
for trace in traces
    accumulate_param_gradients!(trace)
end
```
Finally, we can construct and gradient-based update with [`ParamUpdate`](@ref) and apply it with [`apply!`](@ref).
We can put this all together into a function:
```julia
function train_model(data::Vector{ChoiceMap})
    init_param!(model, :theta, 0.1)
    traces = []
    for observations in data
        trace, = generate(model, model_args, observations)
        push!(traces, trace)
    end
    update = ParamUpdate(FixedStepGradientDescent(0.001), model)
    for iter=1:max_iter
        objective = sum([get_score(trace) for trace in traces])
        println("objective: $objective")
        for trace in traces
            accumulate_param_gradients!(trace)
        end
        apply!(update)
    end
end
```

Note that using the same primitives ([`generate`](@ref) and [`accumulate_param_gradients!`](@ref)), you can compose various more sophisticated learning algorithms involving e.g. stochastic gradient descent and minibatches, and more sophisticated stochastic gradient optimizers like [`ADAM`](@ref).
For example, [`train!`](@ref) trains a generative function from complete data with minibatches.

## Learning from Incomplete Data

When there are random variables in our model whose value is not observed in our data set, then doing maximum learning is significantly more difficult.
Specifically, maximum likelihood is aiming to maximize the **marginal likelihood** of the observed data, which is an integral or sum over the values of the unobserved random variables.
Let's denote the observed variables as `y` and the hidden variables as `z`:
```math
\sum_{i=1}^N \log p(y_i; x, \theta) = \sum_{i=1}^N \log \left( \sum_{z_i} p(z_i, y_i; x, \theta)\right)
```
It is often intractable to evaluate this quantity for specific values of the parameters, let alone maximize it.
Most techniques for learning models from incomplete data, from the EM algorithm to variational autoencoders address this problem by starting with some initial ``\theta = \theta_0`` and iterating between two steps:

- Doing inference about the hidden variables ``z_i`` given the observed variables ``y_i``, for the model with the current values of ``\theta``, which produces some **completions** of the hidden variables ``z_i`` or some representation of the posterior distribution on these hidden variables. This step does not update the parameters ``\theta``.

- Optimize the parameters ``\theta`` to maximize the data of the complete log likelihood, as in the setting of complete data. This step does not involve inference about the hidden variables ``z_i``.

Various algorithms can be understood as examples of this general pattern, although they differ in several details including (i) how they represent the results of inferences, (ii) how they perform the inference step, (iii) whether they try to solve each of the inference and parameter-optimization problems incrementally or not, and (iv) their formal theoretical justification and analysis:

- Expectation maximization (EM) [1], including incremental variants [2]

- Monte Carlo EM [3] and online variants [4]

- Variational EM

- The wake-sleep algorithm [5] and reweighted wake-sleep algorithms [6]

- Variational autoencoders [7]

In Gen, the results of inference are typically represented as a collection of traces of the model, which include values for the latent variables.
The section [Learning from Complete Data](@ref) describes how to perform the parameter update step given a collection of such traces.
In the remainder of this section, we describe various learning algorithms, organized by the inference approach they take to obtain traces.

### Monte Carlo EM

Monte Carlo EM is a broad class of algorithms that use Monte Carlo sampling within the inference step to generate the set of traces that is used for the learning step.
There are many variants possible, based on which Monte Carlo inference algorithm is used.
For example:
```julia
function train_model(data::Vector{ChoiceMap})
    init_param!(model, :theta, 0.1)
    update = ParamUpdate(FixedStepGradientDescent(0.001), model)
    for iter=1:max_iter
        traces = do_monte_carlo_inference(data)
        for trace in traces
            accumulate_param_gradients!(trace)
        end
        apply!(update)
    end
end

function do_monte_carlo_inference(data)
    num_traces = 1000
    (traces, log_weights, _) = importance_sampling(model, (), data, num_samples)
    weights = exp.(log_weights)
    [traces[categorical(weights)] for _=1:num_samples]
end
```
Note that it is also possible to use a weighted collection of traces directly without resampling:
```julia
function train_model(data::Vector{ChoiceMap})
    init_param!(model, :theta, 0.1)
    update = ParamUpdate(FixedStepGradientDescent(0.001), model)
    for iter=1:max_iter
        traces, weights = do_monte_carlo_inference_with_weights(data)
        for (trace, weight) in zip(traces, weights)
            accumulate_param_gradients!(trace, nothing, weight)
        end
        apply!(update)
    end
end
```
MCMC and other algorithms can be used for inference as well.

### Online Monte Carlo EM

The Monte Carlo EM example performed inference from scratch within each iteration.
However, if the change tothe parameters during each iteration is small, it is likely that the traces from the previous iteration can be reused.
There are various ways of reusing traces:
We can use the traces obtained for the previous traces to initialize MCMC for the new parameters.
We can reweight the traces based on the change to their importance weights [4].

### Wake-sleep algorithm

The wake-sleep algorithm [5] is an approach to training generative models that uses an *inference network*, a neural network that takes in the values of observed random variables and returns parameters of a probability distribution on latent variables.
We call the conditional probability distribution on the latent variables, given the observed variables, the *inference model*.
In Gen, both the generative model and the inference model are represented as generative functions.
The wake-sleep algorithm trains the inference model as it trains the generative model.
At each iteration, during the *wake phase*, the generative model is trained on complete traces generated by running the current version of the inference on the observed data.
At each iteration, during the *sleep phase*, the inference model is trained on data generated by simulating from the current generative model.
The [`lecture!`](@ref) or [`lecture_batched!`](@ref) methods can be used for the sleep phase training.

### Reweighted wake-sleep algorithm

The reweighted wake-sleep algorithm [6] is an extension of the wake-sleep algorithm, where during the wake phase, for each observation, a collection of latent completions are taken by simulating from the inference model multiple times.
Then, each of these is weighted by an importance weight.
This extension can be implemented with [`importance_sampling`](@ref).

### Variational inference

Variational inference can be used to for the inference step.
Here, the parameters of the variational approximation, represented as a generative function, are fit to the posterior during the inference step.
[`black_box_vi!`](@ref) or [`black_box_vimco!`](@ref) can be used to fit the variational approximation.
Then, the traces of the model can be obtained by simulating from the variational approximation and merging the resulting choice maps with the observed data.

### Amortized variational inference (VAEs)

Instead of fitting the variational approximation from scratch for each observation, it is possible to fit an *inference model* instead, that takes as input the observation, and generates a distribution on latent variables as output (as in the wake sleep algorithm).
When we train the variational approximation by minimizing the evidence lower bound (ELBO) this is called amortized variational inference.
Variational autencoders are an example.
It is possible to perform amortized variational inference using [`black_box_vi`](@ref) or [`black_box_vimco!`](@ref).

## References

[1] Dempster, Arthur P., Nan M. Laird, and Donald B. Rubin. "Maximum likelihood from incomplete data via the EM algorithm." Journal of the Royal Statistical Society: Series B (Methodological) 39.1 (1977): 1-22. [Link](https://users.fmrib.ox.ac.uk/~jesper/papers/readgroup_070213/DLR_on_EM.pdf)

[2] Neal, Radford M., and Geoffrey E. Hinton. "A view of the EM algorithm that justifies incremental, sparse, and other variants." Learning in graphical models. Springer, Dordrecht, 1998. 355-368. [Link](https://www.cs.toronto.edu/~radford/ftp/emk.pdf)

[3] Wei, Greg CG, and Martin A. Tanner. "A Monte Carlo implementation of the EM algorithm and the poor man's data augmentation algorithms." Journal of the American statistical Association 85.411 (1990): 699-704. [Link](http://www.biostat.jhsph.edu/~rpeng/biostat778/papers/wei-tanner-1990.pdf)

[4] Levine, Richard A., and George Casella. "Implementations of the Monte Carlo EM algorithm." Journal of Computational and Graphical Statistics 10.3 (2001): 422-439. [Link](https://amstat.tandfonline.com/doi/abs/10.1198/106186001317115045)

[5] Hinton, Geoffrey E., et al. "The" wake-sleep" algorithm for unsupervised neural networks." Science 268.5214 (1995): 1158-1161. [Link](https://science.sciencemag.org/content/sci/268/5214/1158.full.pdf)

[6] Jorg Bornschein and Yoshua Bengio. Reweighted wake sleep. ICLR 2015. [Link](https://arxiv.org/pdf/1406.2751.pdf)

[7] Diederik P. Kingma, Max Welling:
Auto-Encoding Variational Bayes. ICLR 2014 [Link](https://arxiv.org/pdf/1312.6114.pdf)

## API

```@docs
lecture!
lecture_batched!
train!
```
