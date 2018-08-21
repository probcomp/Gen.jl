import Distributions

abstract type Distribution{T} end
get_return_type(::Distribution{T}) where {T} = T

export Distribution
export logpdf

#########
# dirac #
#########

struct Dirac <: Distribution{Float64} end

const dirac = Dirac()

function logpdf(::Dirac, x::Real, y::Real)
    x == y ? 0. : -Inf
end

function Base.rand(::Dirac, y::Real)
    y
end

(::Dirac)(y) = rand(Dirac(), y)

get_static_argument_types(::Dirac) = [:Float64]

export dirac

#############
# bernoulli #
#############

struct Bernoulli <: Distribution{Bool} end

const bernoulli = Bernoulli()

function logpdf(::Bernoulli, x::Bool, prob::Real)
    x ? log(prob) : log(1. - prob)
end

Base.rand(::Bernoulli, prob::Real) = rand() < prob

(::Bernoulli)(prob) = rand(Bernoulli(), prob)

get_static_argument_types(::Bernoulli) = [:Float64]

export bernoulli

##########
# normal #
##########

struct Normal <: Distribution{Float64} end

const normal = Normal()

function logpdf(::Normal, x::Real, mu::Real, std::Real)
    var = std * std
    diff = x - mu
    -(diff * diff)/ (2.0 * var) - 0.5 * log(2.0 * pi * var)
end

function Base.rand(::Normal, mu::Real, std::Real)
    mu + std * randn()
end

(::Normal)(mu, std) = rand(Normal(), mu, std)

get_static_argument_types(::Normal) = [:Float64, :Float64]

export normal


#########
# gamma #
#########

struct Gamma <: Distribution{Float64} end

const gamma = Gamma()

function logpdf(::Gamma, x::Real, shape::Real, scale::Real)
    if x > 0.
        (shape - 1.0) * log(x) - (x / scale) - shape * log(scale) - lgamma(shape)
    else
        -Inf
    end
end

function Base.rand(::Gamma, shape::Real, scale::Real)
    rand(Distributions.Gamma(shape, scale))
end

(::Gamma)(shape, scale) = rand(Gamma(), shape, scale)

get_static_argument_types(::Gamma) = [:Float64, :Float64]

export gamma


#################
# inverse gamma #
#################

struct InverseGamma <: Distribution{Float64} end

const inv_gamma = InverseGamma()

function logpdf(::InverseGamma, x::Real, shape::Real, scale::Real)
    if x > 0.
        shape * log(scale) - (shape + 1) * log(x) - lgamma(shape) - (scale / x)
    else
        -Inf
    end
end

function Base.rand(::InverseGamma, shape::Real, scale::Real)
    rand(Distributions.InverseGamma(shape, scale))
end

(::InverseGamma)(shape, scale) = rand(InverseGamma(), shape, scale)

get_static_argument_types(::InverseGamma) = [:Float64, :Float64]

export inv_gamma

########
# beta #
########

struct Beta <: Distribution{Float64} end

const beta = Beta()

function logpdf(::Beta, x::Real, alpha::Real, beta::Real)
    (alpha - 1) * log(x) + (beta - 1) * log1p(-x) - lbeta(alpha, beta)
end

function Base.rand(::Beta, alpha::Real, beta::Real)
    rand(Distributions.Beta(alpha, beta))
end

(::Beta)(alpha, beta) = rand(Beta(), alpha, beta)

get_static_argument_types(::Beta) = [:Float64, :Float64]

export beta

###############
# categorical #
###############

struct Categorical <: Distribution{Int} end

const categorical = Categorical()

function logpdf(::Categorical, x::Int, probs::AbstractArray{U,1}) where {U <: Real}
    log(probs[x])
end

function Base.rand(::Categorical, probs::AbstractArray{U,1}) where {U <: Real}
    rand(Distributions.Categorical(probs))
end

(::Categorical)(probs) = rand(Categorical(), probs)

get_static_argument_types(::Categorical) = [:(Vector{Float64})]

export categorical


####################
# uniform_discrete #
####################

struct UniformDiscrete <: Distribution{Int} end

const uniform_discrete = UniformDiscrete()

function logpdf(::UniformDiscrete, x::Int, low::Integer, high::Integer)
    d = Distributions.DiscreteUniform(low, high)
    Distributions.logpdf(d, x)
end

function Base.rand(::UniformDiscrete, low::Integer, high::Integer)
    rand(Distributions.DiscreteUniform(low, high))
end

(::UniformDiscrete)(low, high) = rand(UniformDiscrete(), low, high)

get_static_argument_types(::UniformDiscrete) = [:Float64, :Float64]

export uniform_discrete


######################
# uniform_continuous #
######################

struct UniformContinuous <: Distribution{Float64} end

const uniform_continuous = UniformContinuous()
const uniform = uniform_continuous

function logpdf(::UniformContinuous, x::Real, low::Real, high::Real)
    (x >= low && x <= high) ? -log(high-low) : -Inf
end

function Base.rand(::UniformContinuous, low::Real, high::Real)
    rand() * (high - low) + low
end

(::UniformContinuous)(low, high) = rand(UniformContinuous(), low, high)

get_static_argument_types(::UniformContinuous) = [:Float64, :Float64]

export uniform_continuous, uniform

###########
# poisson #
###########

struct Poisson <: Distribution{Int} end

const poisson = Poisson()

function logpdf(::Poisson, x::Integer, lambda::Real)
    x * log(lambda) - lambda - lgamma(x+1)
end

function Base.rand(::Poisson, lambda::Real)
    rand(Distributions.Poisson(lambda))
end

(::Poisson)(lambda) = rand(Poisson(), lambda)

get_static_argument_types(::Poisson) = [:Float64]

export poisson
