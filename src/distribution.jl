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


#########
# dirac #
#########

struct Dirac <: Distribution{Float64} end

"""
    dirac(val::Real)

Deterministically return a `Float64` that is equal to the given value.
`logpdf(dirac, x, y)` returns `0` if `x == y` and `-Inf` otherwise.
"""
const dirac = Dirac()

logpdf(::Dirac, x::Real, y::Real) = (x == y ? 0. : -Inf)

random(::Dirac, y::Real) = Float64(y)

(::Dirac)(y) = random(Dirac(), y)

has_output_grad(::Dirac) = false
has_argument_grads(::Dirac) = (false,)
get_static_argument_types(::Dirac) = [Float64]

export dirac


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
get_static_argument_types(::Bernoulli) = [Float64]

export bernoulli

##########
# normal #
##########

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
    deriv_sigma = -0.5 * precision * (1. - precision * (diff * diff))
    (deriv_x, deriv_mu, deriv_sigma)
end

random(::Normal, mu::Real, std::Real) = mu + std * randn()

(::Normal)(mu, std) = random(Normal(), mu, std)

has_output_grad(::Normal) = true
has_argument_grads(::Normal) = (true, true)
get_static_argument_types(::Normal) = [Float64, Float64]

export normal


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
get_static_argument_types(::Gamma) = [Float64, Float64]

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
get_static_argument_types(::InverseGamma) = [Float64, Float64]

export inv_gamma


########
# beta #
########

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
get_static_argument_types(::Beta) = [Float64, Float64]

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
get_static_argument_types(::Categorical) = [Vector{Float64}]

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
get_static_argument_types(::UniformDiscrete) = [Int, Int]

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
get_static_argument_types(::UniformContinuous) = [Float64, Float64]

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
get_static_argument_types(::Poisson) = [Float64]

export poisson
