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
