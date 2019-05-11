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

is_discrete(::InverseGamma) = false

(::InverseGamma)(shape, scale) = random(InverseGamma(), shape, scale)

has_output_grad(::InverseGamma) = false
has_argument_grads(::InverseGamma) = (false, false)

export inv_gamma
