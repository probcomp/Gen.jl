import SpecialFunctions

struct InverseGamma <: Distribution{Float64} end

"""
    inv_gamma(shape::Real, scale::Real)

Sample a `Float64` from a inverse gamma distribution.
"""
const inv_gamma = InverseGamma()

function logpdf(::InverseGamma, x::Real, shape::Real, scale::Real)
    if x > 0.
        shape * log(scale) - (shape + 1) * log(x) - loggamma(shape) - (scale / x)
    else
        -Inf
    end
end

function logpdf_grad(::InverseGamma, x::Real, shape::Real, scale::Real)
    deriv_x = -(shape + 1) / x + scale / (x^2)
    deriv_shape = log(scale) - SpecialFunctions.digamma(shape) - log(x)
    deriv_scale = shape / scale - 1/x
    return (deriv_x, deriv_shape, deriv_scale)
end


function random(::InverseGamma, shape::Real, scale::Real)
    rand(Distributions.InverseGamma(shape, scale))
end

is_discrete(::InverseGamma) = false

(::InverseGamma)(shape, scale) = random(InverseGamma(), shape, scale)

has_output_grad(::InverseGamma) = true
has_argument_grads(::InverseGamma) = (true, true)

export inv_gamma
