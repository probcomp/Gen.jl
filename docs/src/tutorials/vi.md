# [Variational Inference in Gen](@id vi_tutorial)

Variational inference (VI) involves optimizing the parameters of a variational family to maximize a lower bound on the marginal likelihood called the ELBO. In Gen, variational families are represented as generative functions, and variational inference typically involves optimizing the trainable parameters of generative functions.

```@setup vi_tutorial
using Gen, Random
Random.seed!(0)
```

## A Simple Example of VI

Let's begin with a simple example that illustrates how to use Gen's [`black_box_vi!`](@ref) function to perform variational inference. In variational inference, we have a target distribution ``P(x)`` that we wish to approximate with some variational distribution ``Q(x; \phi)`` with trainable parameters ``\phi``.

In many cases, this target distribution is a posterior distribution ``P(x | y)`` given a fixed set of observations ``y``. But in this example, we assume we know ``P(x)`` exactly, and optimize ``\phi`` so that ``Q(x; \phi)`` fits ``P(x)``.

We first define the **target distribution** ``P(x)`` as a normal distribution with 
with a mean of `-1` and a standard deviation of `exp(0.5)`:

```@example vi_tutorial
@gen function target()
    x ~ normal(-1, exp(0.5))
end
nothing # hide
```

We now define a **variational family**, also known as a *guide*, as a generative function ``Q(x; \phi)`` parameterized by a set of trainable parameters ``\phi``. This requires (i) picking the functional form of the variational distribution (e.g. normal, Cauchy, etc.), (ii) choosing how the distribution is parameterized.

Our target distribution is normal, so we make our variational family normally distributed as well. We also define two variational parameters, `x_mu` and `x_log_std`, which are the mean and log standard deviation of our variational distribution.

```@example vi_tutorial
@gen function approx()
    @param x_mu::Float64
    @param x_log_std::Float64
    x ~ normal(x_mu, exp(x_log_std))
end
nothing # hide
```

Since `x_mu` and `x_log_std` are not fixed to particular values, this generative function defines a *family* of distributions, not just one. Note that we intentionally chose to parameterize the distribution by the log standard deviation `x_log_std`, so that every parameter has full support over the real line, and we can perform unconstrained optimization of the parameters.

To perform variational inference, we need to initialize the variational parameters to their starting values:

```@example vi_tutorial
init_param!(approx, :x_mu, 0.0)
init_param!(approx, :x_log_std, 0.0)
nothing # hide
```

Now we can use the [`black_box_vi!`](@ref) function to perform variational inference using [`GradientDescent`](@ref) to update the variational parameters.

```@example vi_tutorial
observations = choicemap()
param_update = ParamUpdate(GradientDescent(1., 1000), approx)
black_box_vi!(target, (), observations, approx, (), param_update;
              iters=200, samples_per_iter=100, verbose=false)
nothing # hide
```

We can now inspect the resulting variational parameters, and see if we have recovered the parameters of the target distribution:

```@example vi_tutorial
x_mu = get_param(approx, :x_mu)
x_log_std = get_param(approx, :x_log_std)
@show x_mu x_log_std;
nothing # hide
```

As expected, we have recovered the parameters of the target distribution.

## Posterior Inference with VI

In the above example, we used a target distribution ``P(x)`` that we had full knowledge about. When performing posterior inference, however, we typically only have the ability to sample from a generative model ``x, y \sim P(x) P(y | x)``, and to evaluate the joint probability ``P(x, y)``, but not the ability to evaluate or sample from the posterior ``P(x | y)`` for a fixed obesrvation ``y``.

Variational inference can address this by approximating ``P(x | y)`` with ``Q(x; \phi)``, allowing us to sample and evaluate ``Q(x; \phi)`` instead. This is done by maximizing a quantity known as the **evidence lower bound** or **ELBO**, which is a lower bound on the log marginal likelihood ``\log P(y)`` of the observations ``y``. The ELBO can be written in multiple equivalent forms:

```math
\begin{aligned}
\operatorname{ELBO}(\phi; y)
&= \mathbb{E}_{x \sim Q(x; \phi)}\left[\log \frac{P(x, y)}{Q(x; \phi)}\right] \\
&= \mathbb{E}_{x \sim Q(x; \phi)}[\log P(x, y)] + \operatorname{H}[Q(x; \phi)] \\
&= \log P(y) - \operatorname{KL}[Q(x; \phi) || P(x | y)]
\end{aligned}
```

Here, ``\operatorname{H}[Q(x; \phi)]`` is the entropy of the variational distribution ``Q(x; \phi)``, and ``\operatorname{KL}[Q(x; \phi) || P(x | y)]`` is the Kullback-Leibler divergence between the variational distribution ``Q(x; \phi)`` and the target distribution ``P(x | y)``. From the third line, we can see that the ELBO is a lower bound on ``\log P(y)``, and that maximizing the ELBO is equivalent to minimizing the KL divergence between ``Q(x; \phi)`` and ``P(x | y)``.

Let's test this for a generative model ``P(x, y)`` where it is possible (with a bit of work) to analytically calculate the posterior ``P(y | x)``:

```@example vi_tutorial
@gen function model(n::Int)
    x ~ normal(0, 1)
    for i in 1:n
        {(:y, i)} ~ normal(x, 0.5)
    end
end
nothing # hide
```

In this normal-normal model, an unknown mean ``x`` is sampled from a ``\operatorname{Normal}(0, 1)`` prior. Then we draw ``n`` datapoints ``y_{1:n}`` from a normal distribution centered around ``x`` with a standard deviation of 0.5. Our task is to infer the posterior distribution over ``x`` given that we have observed ``y_{1:n}``. We'll reuse the same variational family as before:

```@example vi_tutorial
@gen function approx()
    @param x_mu::Float64
    @param x_log_std::Float64
    x ~ normal(x_mu, exp(x_log_std))
end
nothing # hide
```

Suppose we observe ``n = 6`` datapoints ``y_{1:6}`` with the following values:
```@example vi_tutorial
ys = [3.12, 2.25, 2.21, 1.55, 2.15, 1.06]
nothing # hide
```

It is possible to show analytically that the posterior ``P(x | y_{1:n})`` is normally distributed with mean ``\mu_n = \frac{4n}{1 + 4n} \bar y`` and standard deviation ``\sigma_n = \frac{1}{\sqrt{1 + 4n}}``, where ``\bar y`` is the mean of ``y_{1:n}``:

```@example vi_tutorial
n = length(ys)
x_mu_expected = 4*n / (1 + 4*n) * (sum(ys) / n)
x_std_expected = 1/(sqrt((1 + 4*n)))
@show x_mu_expected x_std_expected;
nothing # hide
```

Let's see whether variational inference can reproduce these values. We first construct a choicemap of our observations:

```@example vi_tutorial
observations = choicemap()
for (i, y) in enumerate(ys)
    observations[(:y, i)] = y
end
nothing # hide
```

Next, we configure our [`GradientDescent`](@ref) optimizer. Since this is a more complicated optimization proplem, we use a smaller initial step size of 0.01:

```@example vi_tutorial
step_size_init = 0.01
step_size_beta = 1000
update_config = GradientDescent(step_size_init, step_size_beta)
nothing # hide
```

We then initialize the parameters of our variational approximation, and pass our model, observations, and variational family to [`black_box_vi!`](@ref). 

```@example vi_tutorial
init_param!(approx, :x_mu, 0.0)
init_param!(approx, :x_log_std, 0.0)
param_update = ParamUpdate(update_config, approx);
elbo_est, _, elbo_history =
    black_box_vi!(model, (n,), observations, approx, (), param_update;
                  iters=500, samples_per_iter=200, verbose=false);
nothing # hide
```

As expected, the ELBO estimate increases over time, eventually converging to a value around -9.9:

```@example vi_tutorial
for t in [1; 50:50:500]
    println("iter $(lpad(t, 3)): elbo est. = $(elbo_history[t])")
end
println("final elbo est. = $elbo_est")
```

Inspecting the resulting variational parameters, we find that they are reasonable approximations to the parameters of the true posterior:

```@example vi_tutorial
x_mu_approx = get_param(approx, :x_mu)
Δx_mu = x_mu_approx - x_mu_expected

x_log_std_approx = get_param(approx, :x_log_std)
x_std_approx = exp(x_log_std_approx)
Δx_std = x_std_approx - x_std_expected

@show (x_mu_approx, Δx_mu) (x_std_approx, Δx_std);
nothing # hide
```

## Amortized Variational Inference

In standard variational inference, we have to optimize the variational parameters ``\phi`` for each new inference problem. Depending on how difficult the optimization problem is, this may be costly.

As an alternative, we can perform **amortized variational inference**: Instead of optimizing ``\phi`` for each set of observations ``y`` that we encounter, we learn a *function* ``f_\varphi(y)`` that outputs a set of distribution parameters ``\phi_y`` for each ``y``, and optimize the parameters of the function ``\varphi``. We do this over a dataset of ``K`` independently distributed observation sets ``Y = \{y^1, ..., y^K\}``, maximizing the expected ELBO over this dataset:

```math
\begin{aligned}
\operatorname{A-ELBO}(\varphi; Y)
&= \frac{1}{K} \sum_{k=1}^{K} \operatorname{ELBO}(\varphi; y^k) \\
&= \frac{1}{K} \left[\log P(Y) - \sum_{k=1}^{K} \operatorname{KL}[Q(x; f_{\varphi}(y^k)) || P(x | y^k)] \right]
\end{aligned}
```

We will perform amortized VI over the same generative `model` we defined earlier:

```@example vi_tutorial
@gen function model(n::Int)
    x ~ normal(0, 1)
    for i in 1:n
        {(:y, i)} ~ normal(x, 0.5)
    end
end
nothing # hide
```

Since amortized VI is performed over a dataset of `K` observation sets ``\{y^1, ..., y^K\}``, where each ``y^k`` has ``n`` datapoints ``(y^k_1, ..., y^k_n)`` , we need to nest `model` within a [`Map`](@ref) combinator that repeats `model` ``K`` times:

```@example vi_tutorial
mapped_model = Map(model)
nothing # hide
```

Let's generate a synthetic dataset of ``K = 10`` observation sets, each with ``n = 6`` datapoints:

```@example vi_tutorial
# Simulate 10 observation sets of length 6
K, n = 10, 6
mapped_trace = simulate(mapped_model, (fill(n, K),))
observations = get_choices(mapped_trace)

# Select just the `y` values, excluding the generated `x` values
sel = select((k => (:y, i) for i in 1:n for k in 1:K)...)
observations = get_selected(observations, sel)
all_ys = [[observations[k => (:y, i)] for i in 1:n] for k in 1:K]
nothing # hide
```

Now let's define our amortized approximation, which takes in an observation set `ys`, and computes the parameters of a normal distribution over `x` as a function of `ys`:

```@example vi_tutorial
@gen function amortized_approx(ys)
    @param x_mu_bias::Float64
    @param x_mu_coeff::Float64
    @param x_log_std::Float64
    x_mu = x_mu_bias + x_mu_coeff * sum(ys)
    x ~ normal(x_mu, exp(x_log_std))
    return (x_mu, x_log_std)
end
nothing # hide
```

Similar to our `model`, we need to wrap this variational approximation in a [`Map`](@ref) combinator:

```@example vi_tutorial
mapped_approx = Map(amortized_approx)
nothing # hide
```

In our choice of function ``f_\varphi(y)``, we exploit the fact that the posterior mean `x_mu` should depend on the sum of the values in `ys`, along with the knowledge that `x_log_std` does not depend on `ys`. We could have chosen a more complex function, such as full-rank linear regression, or a neural network, but this would make optimization more difficult. Given this choice of function, the optimal parameters ``\varphi^*`` can be computed analytically:

```@example vi_tutorial
n = 6

x_mu_bias_optimal = 0.0
x_mu_coeff_optimal = 4 / (1 + 4*n)

x_std_optimal = 1/(sqrt((1 + 4*n)))
x_log_std_optimal = log(x_std_optimal)

@show x_mu_bias_optimal x_mu_coeff_optimal x_log_std_optimal;
nothing # hide
```

We can now fit our variational approximation via [`black_box_vi!`](@ref): We initialize the variational parameters, then configure our parameter update to update the parameters of `amortized_approx`:

```@example vi_tutorial
# Configure parameter update to optimize the parameters of `amortized_approx`
step_size_init = 1e-4
step_size_beta = 1000
update_config = GradientDescent(step_size_init, step_size_beta)

# Initialize the amortized variational parameters, then the parameter update
init_param!(amortized_approx, :x_mu_bias, 0.0);
init_param!(amortized_approx, :x_mu_coeff, 0.0);
init_param!(amortized_approx, :x_log_std, 0.0);
param_update = ParamUpdate(update_config, amortized_approx);

# Run amortized black-box variational inference over the synthetic observations
mapped_model_args = (fill(n, K), )
mapped_approx_args = (all_ys, )
elbo_est, _, elbo_history =
    black_box_vi!(mapped_model, mapped_model_args, observations,
                  mapped_approx, mapped_approx_args, param_update;
                  iters=500, samples_per_iter=100, verbose=false);
nothing # hide
```

Once again, the ELBO estimate increases and eventually converges:

```@example vi_tutorial
for t in [1; 50:50:500]
    println("iter $(lpad(t, 3)): elbo est. = $(elbo_history[t])")
end
println("final elbo est. = $elbo_est")
```

Our amortized variational parameters ``\varphi`` are also fairly close to their optimal values ``\varphi^*``:

```@example vi_tutorial
x_mu_bias = get_param(amortized_approx, :x_mu_bias)
Δx_mu_bias = x_mu_bias - x_mu_bias_optimal

x_mu_coeff = get_param(amortized_approx, :x_mu_coeff)
Δx_mu_coeff = x_mu_coeff - x_mu_coeff_optimal

x_log_std = get_param(amortized_approx, :x_log_std)
Δx_log_std = x_log_std - x_log_std_optimal

@show (x_mu_bias, Δx_mu_bias) (x_mu_coeff, Δx_mu_coeff) (x_log_std, Δx_log_std);
nothing # hide
```

If we now call `amortized_approx` with our observation set `ys` from the previous section, we should get something close to what standard variational inference produced by optimizing the paramaters of `approx` directly: 

```@example vi_tutorial
x_mu_amortized, x_log_std_amortized = amortized_approx(ys)
x_std_amortized = exp(x_log_std_amortized)

@show x_mu_amortized x_std_amortized;
@show x_mu_approx x_std_approx;
@show x_mu_expected x_std_expected;
nothing # hide
```

Both amortized VI and standard VI produce parameter estimates that are reasonably close to the paramters of the true posterior.

## Reparametrization Trick

To use the reparametrization trick to reduce the variance of gradient estimators, users currently need to write two versions of their variational family, one that is reparametrized and one that is not. Gen.jl does not currently include inference library support for this. We plan to add automated support for reparametrization and other variance reduction techniques in the future.
