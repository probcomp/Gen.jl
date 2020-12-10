##################################################################
# homogeneous mixture: arbitrary number of the same distribution #
##################################################################

"""
    HomogeneousMixture(distribution::Distribution, dims::Vector{Int})

Defines a new type for a mixture distribution family.

The first argument defines the base distribution family of each component in the mixture.

The second argument must have length equal
to the number of arguments taken by the base distribution family.  A value of 0
at a position in the vector an indicates that the corresponding argument to the
base distribution is a scalar, and integer values of i for i >= 1 indicate that
the corresponding argument is an i-dimensional array.

Example:

    mixture_of_normals = HomogeneousMixture(normal, [0, 0])

The resulting distribution (e.g. `mixture_of_normals` above) can then be used like the built-in distribution values like `normal`.
The distribution takes `n+1` arguments where `n` is the number of arguments taken by the base distribution.
The first argument to the distribution is a vector of non-negative mixture weights, which must sum to 1.0.
The remaining arguments to the distribution correspond to the arguments of the base distribution, but have a different type:
If an argument to the base distribution is a scalar of type `T`, then the corresponding argument to the mixture distribution is a `Vector{T}`, where each element of this vector is the argument to the corresponding mixture component.
If an argument to the base distribution is an `Array{T,N}` for some `N`, then the corresponding argument to the mixture distribution is of the form `arr::Array{T,N+1}`, where each slice of the array of the form `arr[:,:,...,i]` is the argument for the `i`th mixture component.

Example:

    mixture_of_normals = HomogeneousMixture(normal, [0, 0])
    mixture_of_mvnormals = HomogeneousMixture(mvnormal, [1, 2])

    @gen function foo()
        # mixture of two normal distributions
        # with means -1.0 and 1.0
        # and standard deviations 0.1 and 10.0
        # the first normal distribution has weight 0.4 and the second has weight 0.6
        x ~ mixture_of_normals([0.4, 0.6], [-1.0, 1.0], [0.1, 10.0])

        # mixture of two multivariate normal distributions
        # with means: [0.0, 0.0] and [1.0, 1.0]
        # and covariance matrices: [1.0 0.0; 0.0 1.0] and [10.0 0.0; 0.0 10.0]
        # the first multivariate normal distribution has weight 0.4 and the second has weight 0.6
        means = [0.0 1.0; 0.0 1.0] # or, cat([0.0, 0.0], [1.0, 1.0], dims=2)
        covs = cat([1.0 0.0; 0.0 1.0], [10.0 0.0; 0.0 10.0], dims=3)
        y ~ mixture_of_mvnormals([0.4, 0.6], means, covs)
    end
"""
struct HomogeneousMixture{T} <: Distribution{T}
    base_dist::Distribution{T}
    dims::Vector{Int}
end

(dist::HomogeneousMixture)(args...) = random(dist, args...)

Gen.has_output_grad(dist::HomogeneousMixture) = has_output_grad(dist.base_dist)
Gen.has_argument_grads(dist::HomogeneousMixture) = (true, has_argument_grads(dist.base_dist)...)
Gen.is_discrete(dist::HomogeneousMixture) = is_discrete(dist.base_dist)

function args_for_component(dist::HomogeneousMixture, k::Int, args)
    # returns a generator
    return (arg[fill(Colon(), dim)..., k]
            for (arg, dim) in zip(args, dist.dims))
end

function Gen.random(dist::HomogeneousMixture, weights, args...)
    k = categorical(weights)
    return random(dist.base_dist, args_for_component(dist, k, args)...)
end

function Gen.logpdf(dist::HomogeneousMixture, x, weights, args...)
    K = length(weights)
    log_densities = [Gen.logpdf(dist.base_dist, x, args_for_component(dist, k, args)...) for k in 1:K]
    log_densities = log_densities .+ log.(weights)
    return logsumexp(log_densities)
end

function Gen.logpdf_grad(dist::HomogeneousMixture, x, weights, args...)
    K = length(weights)
    log_densities = [Gen.logpdf(dist.base_dist, x, args_for_component(dist, k, args)...) for k in 1:K]
    log_weighted_densities = log_densities .+ log.(weights)
    relative_weighted_densities = exp.(log_weighted_densities .- logsumexp(log_weighted_densities))

    # log_grads[k] contains the gradients for the k'th component
    log_grads = [Gen.logpdf_grad(dist.base_dist, x, args_for_component(dist, k, args)...) for k in 1:K]

    # compute gradient with respect to x
    log_grads_x = [log_grad[1] for log_grad in log_grads]
    x_grad = if has_output_grad(dist.base_dist)
        sum(log_grads_x .* relative_weighted_densities)
    else
        nothing
    end

    # compute gradients with respect to the weights
    weights_grad = exp.(log_densities .- logsumexp(log_weighted_densities))

    # compute gradients with respect to each argument
    arg_grads = Any[]
    for (i, (has_grad, arg, dim)) in enumerate(zip(has_argument_grads(dist)[2:end], args, dist.dims))
        if has_grad
            if dim == 0
                grads = [log_grad[i+1] for log_grad in log_grads]
                grad_weights = relative_weighted_densities
            else
                grads = cat(
                    [log_grad[i+1] for log_grad in log_grads]...,
                    dims=dist.dims[i]+1)
                grad_weights = reshape(
                    relative_weighted_densities,
                    (1 for d in 1:dist.dims[i])..., length(dist.dims))
            end
            push!(arg_grads, grads .* grad_weights)
        else
            push!(arg_grads, nothing)
        end
    end

    return (x_grad, weights_grad, arg_grads...)
end

export HomogeneousMixture


##############################################################################
# heterogeneous mixture: fixed number of potentially different distributions #
##############################################################################

"""
    HeterogeneousMixture(distributions::Vector{Distribution{T}}) where {T}

Defines a new mixture distribution family.

The argument is the vector of base distributions, one for each mixture component.

Note that the base distributions must have the same output type.

Example:

    uniform_beta_mixture = HeterogeneousMixture([uniform, beta])

The resulting mixture distribution takes `n+1` arguments, where `n` is the sum of the number of arguments taken by each distribution in the list.
The first argument to the mixture distribution is a vector of non-negative mixture weights, which must sum to 1.0.
The remaining arguments are the arguments to each mixture component distribution, in order in which the distributions are passed into the constructor.

Example:

    @gen function foo()
        # mixure of a uniform distribution on the interval [`lower`, `upper`]
        # and a beta distribution with alpha parameter `a` and beta parameter `b`
        # the uniform as weight 0.4 and the beta has weight 0.6
        x ~ uniform_beta_mixture([0.4, 0.6], lower, upper, a, b)
    end
"""
struct HeterogeneousMixture{T} <: Distribution{T}
    K::Int
    distributions::Vector{Distribution{T}}
    has_output_grad::Bool
    has_argument_grads::Tuple
    is_discrete::Bool
    num_args::Vector{Int}
    starting_args::Vector{Int}
end

(dist::HeterogeneousMixture)(args...) = random(dist, args...)

Gen.has_output_grad(dist::HeterogeneousMixture) = dist.has_output_grad
Gen.has_argument_grads(dist::HeterogeneousMixture) = dist.has_argument_grads
Gen.is_discrete(dist::HeterogeneousMixture) = dist.is_discrete

const MIXTURE_WRONG_NUM_COMPONENTS_ERR = "the length of the weights vector does not match the number of mixture components"

function HeterogeneousMixture(distributions::Vector{Distribution{T}}) where {T}
    _has_output_grad = true
    _has_argument_grads = Bool[true] # weights
    _is_discrete = true
    for dist in distributions
        _has_output_grad = _has_output_grad && has_output_grad(dist)
        for has_arg_grad in has_argument_grads(dist)
            push!(_has_argument_grads, has_arg_grad)
        end
        _is_discrete = _is_discrete && is_discrete(dist)
    end
    num_args = Int[]
    starting_args = Int[]
    for dist in distributions
        push!(starting_args, sum(num_args) + 1)
        push!(num_args, length(has_argument_grads(dist)))
    end
    K = length(distributions)
    return HeterogeneousMixture{T}(
        K, distributions,
        _has_output_grad,
        tuple(_has_argument_grads...),
        _is_discrete,
        num_args,
        starting_args)
end

function extract_args_for_component(dist::HeterogeneousMixture, component_args_flat, k::Int)
    start_arg = dist.starting_args[k]
    n = dist.num_args[k]
    return component_args_flat[start_arg:start_arg+n-1]
end

function Gen.random(dist::HeterogeneousMixture{T}, weights, component_args_flat...) where {T}
    (length(weights) != dist.K) && error(MIXTURE_WRONG_NUM_COMPONENTS_ERR)
    k = categorical(weights)
    value::T = random(
        dist.distributions[k],
        extract_args_for_component(dist, component_args_flat, k)...)
    return value
end

function Gen.logpdf(dist::HeterogeneousMixture, x, weights, component_args_flat...)
    (length(weights) != dist.K) && error(MIXTURE_WRONG_NUM_COMPONENTS_ERR)
    log_densities = [Gen.logpdf(
            dist.distributions[k], x,
            extract_args_for_component(dist, component_args_flat, k)...)
        for k in 1:dist.K]
    log_densities = log_densities .+ log.(weights)
    return logsumexp(log_densities)
end

function Gen.logpdf_grad(dist::HeterogeneousMixture, x, weights, component_args_flat...)
    (length(weights) != dist.K) && error(MIXTURE_WRONG_NUM_COMPONENTS_ERR)
    log_densities = [Gen.logpdf(
            dist.distributions[k], x,
            extract_args_for_component(dist, component_args_flat, k)...)
        for k in 1:dist.K]
    log_weighted_densities = log_densities .+ log.(weights)
    relative_weighted_densities = exp.(log_weighted_densities .- logsumexp(log_weighted_densities))

    # log_grads[k] contains the gradients for that k in the mixture
    log_grads = [Gen.logpdf_grad(
            dist.distributions[k], x,
            extract_args_for_component(dist, component_args_flat, k)...)
        for k in 1:dist.K]

    # gradient with respect to x
    log_grads_x = [log_grad[1] for log_grad in log_grads]
    x_grad = if has_output_grad(dist)
        sum(log_grads_x .* relative_weighted_densities)
    else
        nothing
    end

    # gradients with respect to the weights
    weights_grad = exp.(log_densities .- logsumexp(log_weighted_densities))

    # gradients with respect to each argument of each component
    component_arg_grads = Any[]
    cur = 1
    for k in 1:dist.K
        for i in 1:dist.num_args[k]
            if dist.has_argument_grads[cur]
                @assert log_grads[k][i+1] != nothing
                push!(component_arg_grads, relative_weighted_densities[k] * log_grads[k][i+1])
            else
                @assert log_grads[k][i+1] == nothing
                push!(component_arg_grads, nothing)
            end
            cur += 1
        end
    end
    
    return (x_grad, weights_grad, component_arg_grads...)
end

export HeterogeneousMixture
