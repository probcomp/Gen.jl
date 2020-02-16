# Learning Generative Functions

Learning and inference are closely related concepts, and the distinction between the two is not always clear.
Often, **learning** refers to inferring long-lived unobserved quantities that will be reused across many problem instances (like a dynamics model for an entity that we are trying to track), whereas **inference** refers to inferring shorter-lived quantities (like a specific trajectory of a specific entity).
Learning is the way to use data to automatically generate **models** of the world, or to automatically fill in unknown parameters in hand-coded models.
These resulting models are then used in various inference tasks.

There are many variants of the learning task---we could be training the weights of a neural network, estimating a handful of parameters in a structured and hand-coded model, or we could be learning the structure or architecture of a model.
Also, we could do Bayesian learning in which we seek a probability distribution on possible models, or we could seek just the best model, as measured by e.g. **maximum likelihood**.
This section focuses on maximum likelihood learning of the [Trainable parameters] of a generative function.
These are numerical quantities that are part of the generative function's state, with respect to which generative functions are able to report gradients of their (log) probability density function of their density function.
Trainable parameters are different from random choices --- random choices are per-trace and trainable parameters are a property of the generative function (which is associated with many traces).
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
    update = ParamUpdate(FixedStepSizeGradientDescent(0.001), model)
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

- [Expectation maximization (EM) [1], including incremental variants [2]

- Monte Carlo EM [3]

- Variational EM

- The wake-sleep algorithm [5] and reweighted wake-sleep algorithms [6]

- Variational autoencoders [7]

In Gen, the results of inference are typically represented as a collection of traces of the model, which include values for the latent variables.
The section [Learning from Complete Data](@ref) describes how to perform the parameter update step given a collection of such traces.
In the remainder of this section, we describe various learning algorithms, organized various approaches obtaining these traces via different forms of probabilistic inference.

### Monte Carlo EM

### Online Monte Carlo EM

https://www.tandfonline.com/doi/pdf/10.1198/106186001317115045?needAccess=true

### Wake-sleep algorithm

The wake-sleep algorithm is an approach to training generative models, in which an *inference network*

### Reweighted wake-sleep algorithm

### Variational infernece

### Amortized variational inference (VAEs)






## References

[1] Dempster, Arthur P., Nan M. Laird, and Donald B. Rubin. "Maximum likelihood from incomplete data via the EM algorithm." Journal of the Royal Statistical Society: Series B (Methodological) 39.1 (1977): 1-22. [Link](https://users.fmrib.ox.ac.uk/~jesper/papers/readgroup_070213/DLR_on_EM.pdf)

[2] Neal, Radford M., and Geoffrey E. Hinton. "A view of the EM algorithm that justifies incremental, sparse, and other variants." Learning in graphical models. Springer, Dordrecht, 1998. 355-368. [Link](https://www.cs.toronto.edu/~radford/ftp/emk.pdf)

[3] Wei, Greg CG, and Martin A. Tanner. "A Monte Carlo implementation of the EM algorithm and the poor man's data augmentation algorithms." Journal of the American statistical Association 85.411 (1990): 699-704. [Link](http://www.biostat.jhsph.edu/~rpeng/biostat778/papers/wei-tanner-1990.pdf)

[5] Hinton, Geoffrey E., et al. "The" wake-sleep" algorithm for unsupervised neural networks." Science 268.5214 (1995): 1158-1161. [Link](https://science.sciencemag.org/content/sci/268/5214/1158.full.pdf)

[5] Jorg Bornschein and Yoshua Bengio. Reweighted wake sleep. ICLR 2015. [Link](https://arxiv.org/pdf/1406.2751.pdf)

[7] Diederik P. Kingma, Max Welling:
Auto-Encoding Variational Bayes. ICLR 2014 [Link](https://arxiv.org/pdf/1312.6114.pdf)
