import Distributions
using SpecialFunctions: lgamma, lbeta, digamma


#########################
# abstract distribution #
#########################

abstract type Distribution{T} end

function random end
function logpdf end
get_return_type(::Distribution{T}) where {T} = T

export Distribution
export random
export logpdf
export logpdf_grad


#############
# bernoulli #
#############

struct Bernoulli <: Distribution{Bool} end

"""
    bernoulli(prob_true::Real)

Samples a `Bool` value which is true with given probability
"""
const bernoulli = Bernoulli()

function logpdf(::Bernoulli, x::Bool, prob::Real)
    x ? log(prob) : log(1. - prob)
end

function logpdf_grad(::Bernoulli, x::Bool, prob::Real)
    prob_grad = x ? 1. / prob : -1. / (1-prob)
    (nothing, prob_grad)
end

random(::Bernoulli, prob::Real) = rand() < prob

(::Bernoulli)(prob) = random(Bernoulli(), prob)

has_output_grad(::Bernoulli) = false
has_argument_grads(::Bernoulli) = (true,)

export bernoulli

#####################
# univariate normal #
#####################

struct Normal <: Distribution{Float64} end

"""
    normal(mu::Real, std::Real)

Samples a `Float64` value from a normal distribution.
"""
const normal = Normal()

function logpdf(::Normal, x::Real, mu::Real, std::Real)
    var = std * std
    diff = x - mu
    -(diff * diff)/ (2.0 * var) - 0.5 * log(2.0 * pi * var)
end

function logpdf_grad(::Normal, x::Real, mu::Real, std::Real)
    precision = 1. / (std * std)
    diff = mu - x
    deriv_x = diff * precision
    deriv_mu = -deriv_x
    deriv_std = -1. / std + (diff * diff) / (std * std * std)
    (deriv_x, deriv_mu, deriv_std)
end

random(::Normal, mu::Real, std::Real) = mu + std * randn()

(::Normal)(mu, std) = random(Normal(), mu, std)

has_output_grad(::Normal) = true
has_argument_grads(::Normal) = (true, true)

export normal

#######################
# multivariate normal #
#######################

struct MultivariateNormal <: Distribution{Vector{Float64}} end

"""
    mvnormal(mu::AbstractVector{T}, cov::AbstractMatrix{U}} where {T<:Real,U<:Real}

Samples a `Vector{Float64}` value from a multivariate normal distribution.
"""
const mvnormal = MultivariateNormal()

function logpdf(::MultivariateNormal, x::AbstractVector{T}, mu::AbstractVector{U},
                cov::AbstractMatrix{V}) where {T,U,V}
    dist = Distributions.MvNormal(mu, cov)
    Distributions.logpdf(dist, x)
end

function logpdf_grad(::Normal, x::AbstractVector{T}, mu::AbstractVector{U},
                cov::AbstractMatrix{V}) where {T,U,V}
    dist = Distributions.MvNormal(mu, cov)
    x_deriv = Distributions.gradlogpdf(dist, x)
    (x_deriv, nothing, nothing)
end

function random(::MultivariateNormal, mu::AbstractVector{U},
                cov::AbstractMatrix{V}) where {T,U,V}
    rand(Distributions.MvNormal(mu, cov))
end

(::MultivariateNormal)(mu, cov) = random(MultivariateNormal(), mu, cov)

has_output_grad(::MultivariateNormal) = true
has_argument_grads(::MultivariateNormal) = (false, false)

export mvnormal

#########
# gamma #
#########

struct Gamma <: Distribution{Float64} end

"""
    gamma(shape::Real, scale::Real)

Sample a `Float64` from a gamma distribution.
"""
const gamma = Gamma()

function logpdf(::Gamma, x::Real, shape::Real, scale::Real)
    if x > 0.
        (shape - 1.0) * log(x) - (x / scale) - shape * log(scale) - lgamma(shape)
    else
        -Inf
    end
end

function logpdf_grad(::Gamma, x::Real, shape::Real, scale::Real)
    error("Not Implemented")
    (nothing, nothing, nothing)
end

function random(::Gamma, shape::Real, scale::Real)
    rand(Distributions.Gamma(shape, scale))
end

(::Gamma)(shape, scale) = random(Gamma(), shape, scale)

has_output_grad(::Gamma) = false
has_argument_grads(::Gamma) = (false, false)

export gamma


#################
# inverse gamma #
#################

struct InverseGamma <: Distribution{Float64} end

"""
    inv_gamma(shape::Real, scale::Real)

Sample a `Float64` from a inverse gamma distribution.
"""
const inv_gamma = InverseGamma()

function logpdf(::InverseGamma, x::Real, shape::Real, scale::Real)
    if x > 0.
        shape * log(scale) - (shape + 1) * log(x) - lgamma(shape) - (scale / x)
    else
        -Inf
    end
end

function logpdf_grad(::InverseGamma, x::Real, shape::Real, scale::Real)
    error("Not Implemented")
    (nothing, nothing, nothing)
end


function random(::InverseGamma, shape::Real, scale::Real)
    rand(Distributions.InverseGamma(shape, scale))
end

(::InverseGamma)(shape, scale) = random(InverseGamma(), shape, scale)

has_output_grad(::InverseGamma) = false
has_argument_grads(::InverseGamma) = (false, false)

export inv_gamma


########
# beta #
########

# TODO allow the lower and upper bounds to be parameterized.
# use default values for the lower and upper bounds ? (0 and 1)

struct Beta <: Distribution{Float64} end

"""
    beta(alpha::Real, beta::Real)

Sample a `Float64` from a beta distribution.
"""
const beta = Beta()

function logpdf(::Beta, x::Real, alpha::Real, beta::Real)
    (alpha - 1) * log(x) + (beta - 1) * log1p(-x) - lbeta(alpha, beta)
end

function logpdf_grad(::Beta, x::Real, alpha::Real, beta::Real)
    if 0 <= x <= 1
        deriv_x = (alpha - 1) / x - (beta - 1) / (1 - x)
        deriv_alpha = log(x) - (digamma(alpha) - digamma(alpha + beta))
        deriv_beta = log1p(-x) - (digamma(beta) - digamma(alpha + beta))
    else
        error("x is outside of support: $x")
    end
    (deriv_x, deriv_alpha, deriv_beta)
end

function random(::Beta, alpha::Real, beta::Real)
    rand(Distributions.Beta(alpha, beta))
end

(::Beta)(alpha, beta) = random(Beta(), alpha, beta)

has_output_grad(::Beta) = true
has_argument_grads(::Beta) = (true, true)

export beta


###############
# categorical #
###############

struct Categorical <: Distribution{Int} end

"""
    categorical(probs::AbstractArray{U, 1}) where {U <: Real}

Given a vector of probabilities `probs` where `sum(probs) = 1`, sample an `Int` `i` from the set {1, 2, .., `length(probs)`} with probability `probs[i]`.
"""
const categorical = Categorical()

function logpdf(::Categorical, x::Int, probs::AbstractArray{U,1}) where {U <: Real}
    log(probs[x])
end

function logpdf_grad(::Beta, x::Int, probs::AbstractArray{U,1})  where {U <: Real}
    grad = zeros(length(probs))
    grad[x] = 1.0
    (nothing, grad)
end

function random(::Categorical, probs::AbstractArray{U,1}) where {U <: Real}
    rand(Distributions.Categorical(probs))
end

(::Categorical)(probs) = random(Categorical(), probs)

has_output_grad(::Categorical) = false
has_argument_grads(::Categorical) = (true,)

export categorical


####################
# uniform_discrete #
####################

struct UniformDiscrete <: Distribution{Int} end

"""
    uniform_discrete(low::Integer, high::Integer)

Sample an `Int` from the uniform distribution on the set {low, low + 1, ..., high-1, high}.
"""
const uniform_discrete = UniformDiscrete()

function logpdf(::UniformDiscrete, x::Int, low::Integer, high::Integer)
    d = Distributions.DiscreteUniform(low, high)
    Distributions.logpdf(d, x)
end

function logpdf_grad(::UniformDiscrete, x::Int, lower::Integer, high::Integer)
    (nothing, nothing, nothing)
end

function random(::UniformDiscrete, low::Integer, high::Integer)
    rand(Distributions.DiscreteUniform(low, high))
end

(::UniformDiscrete)(low, high) = random(UniformDiscrete(), low, high)

has_output_grad(::UniformDiscrete) = false
has_argument_grads(::UniformDiscrete) = (false, false)

export uniform_discrete


######################
# uniform_continuous #
######################

struct UniformContinuous <: Distribution{Float64} end

const uniform_continuous = UniformContinuous()

"""
    uniform(low::Real, high::Real)

Sample a `Float64` from the uniform distribution on the interval [low, high].
"""
const uniform = uniform_continuous

function logpdf(::UniformContinuous, x::Real, low::Real, high::Real)
    (x >= low && x <= high) ? -log(high-low) : -Inf
end

function logpdf_grad(::UniformContinuous, x::Real, low::Real, high::Real)
    inv_diff = 1. / (high-low)
    (0., inv_diff, -inv_diff)
end

function random(::UniformContinuous, low::Real, high::Real)
    rand() * (high - low) + low
end

(::UniformContinuous)(low, high) = random(UniformContinuous(), low, high)

has_output_grad(::UniformContinuous) = true
has_argument_grads(::UniformContinuous) = (true, true)

export uniform_continuous, uniform


###########
# poisson #
###########

struct Poisson <: Distribution{Int} end

"""
    poisson(lambda::Real)

Sample an `Int` from the Poisson distribution with rate `lambda`.
"""
const poisson = Poisson()

function logpdf(::Poisson, x::Integer, lambda::Real)
    x * log(lambda) - lambda - lgamma(x+1)
end

function logpdf_grad(::Poisson, x::Integer, lambda::Real)
    error("Not implemented")
    (nothing, nothing)
end


function random(::Poisson, lambda::Real)
    rand(Distributions.Poisson(lambda))
end

(::Poisson)(lambda) = random(Poisson(), lambda)

has_output_grad(::Poisson) = false
has_argument_grads(::Poisson) = (false,)

export poisson


#####################
# Piecewise Uniform #
#####################

struct PiecewiseUniform <: Distribution{Float64} end

"""
    piecewise_uniform(bounds, probs)

Samples a `Float64` value from a piecewise uniform continuous distribution.

There are `n` bins where `n = length(probs)` and `n + 1 = length(bounds)`.
Bounds must satisfy `bounds[i] < bounds[i+1]` for all `i`.
The probability density at `x` is zero if `x <= bounds[1]` or `x >= bounds[end]` and is otherwise `probs[bin] / (bounds[bin] - bounds[bin+1])` where `bounds[bin] < x <= bounds[bin+1]`.
"""
const piecewise_uniform = PiecewiseUniform()

function check_dims(::PiecewiseUniform, bounds, probs)
    if length(bounds) != length(probs) + 1
        error("Dimension mismatch")
    end
end

function get_bin(bounds, x)
    @assert x <= bounds[end]
    bin = 1
    while x > bounds[bin+1]
        bin += 1
    end
    @assert x > bounds[bin] && x <= bounds[bin+1]
    bin
end

function logpdf(::PiecewiseUniform, x::Real, bounds::AbstractVector{T},
                    probs::AbstractVector{U}) where {T <: Real, U <: Real}
    check_dims(piecewise_uniform, bounds, probs)

    # bounds[1]      bounds[2]           bounds[3]      bounds[4]
    # ^              ^                   ^              ^
    # |    probs[1]  |  probs[2]         | probs[3]     |
    if x <= bounds[1] || x >= bounds[end]
        -Inf
    end
    bin = get_bin(bounds, x)
    log(probs[bin]) - log(bounds[bin+1] - bounds[bin])
end

function random(::PiecewiseUniform, bounds::Vector{T},
                    probs::Vector{U}) where {T <: Real, U <: Real}
    bin = categorical(probs)
    uniform_continuous(bounds[bin], bounds[bin+1])
end

function logpdf_grad(::PiecewiseUniform, x::Real, bounds, probs)
    check_dims(piecewise_uniform, bounds, probs)
    if x <= bounds[1] || x >= bounds[end]
        error("Out of bounds")
    end
    bin = get_bin(bounds, x)
    bounds_grad = fill(0., length(bounds))
    bin_length = bounds[bin+1] - bounds[bin]
    bounds_grad[bin] = 1. / bin_length
    bounds_grad[bin+1] = - 1. / bin_length
    probs_grad = fill(0., length(probs))
    probs_grad[bin] = 1. / probs[bin]
    (0., bounds_grad, probs_grad)
end

has_output_grad(::PiecewiseUniform) = true
has_argument_grads(::PiecewiseUniform) = (true, true)

export piecewise_uniform

########################
# beta-uniform mixture #
########################

# TODO allow the lower and upper bounds to be changed, like uniform.

struct BetaUniformMixture <: Distribution{Float64} end

"""
    beta_uniform(theta::Real, alpha::Real, beta::Real)

Samples a `Float64` value from a mixture of a uniform distribution on [0, 1] with probability `1-theta` and a beta distribution with parameters `alpha` and `beta` with probability `theta`.
"""
const beta_uniform = BetaUniformMixture()

function logpdf(::BetaUniformMixture, x::Real, theta::Real, alpha::Real, beta::Real)
    lbeta = log(theta) + logpdf(Beta(), x, alpha, beta)
    luniform = log(1.0 - theta)
    logsumexp(lbeta, luniform)
end

function logpdf_grad(::BetaUniformMixture, x::Real, theta::Real, alpha::Real, beta::Real)
    beta_logpdf = logpdf(Beta(), x, alpha, beta)
    uniform_logpdf = logpdf(uniform_continuous, x, 0., 1.)
    beta_grad = logpdf_grad(Beta(), x, alpha, beta)
    uniform_grad = logpdf_grad(uniform_continuous, x, 0., 1.)
    w1 = 1. / (1. + exp(log(1. - theta) + uniform_logpdf - log(theta) - beta_logpdf))
    w2 = 1. - w1
    x_deriv = w1 * beta_grad[1] + w2 * uniform_grad[1]
    alpha_deriv = w1 * beta_grad[2]
    beta_deriv = w1 * beta_grad[3]
    theta_deriv = (exp(beta_logpdf) - exp(uniform_logpdf)) / (theta * exp(beta_logpdf) + (1. - theta) * exp(uniform_logpdf))
    (x_deriv, theta_deriv, alpha_deriv, beta_deriv)
end

function random(::BetaUniformMixture, theta::Real, alpha::Real, beta::Real)
    if bernoulli(theta)
        random(Beta(), alpha, beta)
    else
        random(uniform_continuous, 0., 1.)
    end
end

(::BetaUniformMixture)(theta, alpha, beta) = random(BetaUniformMixture(), theta, alpha, beta)

has_output_grad(::BetaUniformMixture) = true
has_argument_grads(::BetaUniformMixture) = (true, true, true)

export beta_uniform
