struct Gamma <: Distribution{Float64} end

"""
    gamma(shape::Real, scale::Real)

Sample a `Float64` from a gamma distribution.
"""
const gamma = Gamma()

function logpdf(::Gamma, x::Real, shape::Real, scale::Real)
    if x > 0.
        (shape - 1.0) * log(x) - (x / scale) - shape * log(scale) - loggamma(shape)
    else
        -Inf
    end
end

function logpdf_grad(::Gamma, x::Real, shape::Real, scale::Real)
    if x > 0.
        deriv_x = (shape - 1.) / x - (1. / scale)
        deriv_shape = log(x) - log(scale) - digamma(shape)
        deriv_scale = x / (scale * scale) - (shape / scale)
        (deriv_x, deriv_shape, deriv_scale)
    else
        (0., 0., 0.)
    end
end

function random(rng::AbstractRNG, ::Gamma, shape::Real, scale::Real)
    rand(rng, Distributions.Gamma(shape, scale))
end

is_discrete(::Gamma) = false

(dist::Gamma)(shape, scale) = dist(default_rng(), shape, scale)
(::Gamma)(rng::AbstractRNG, shape, scale) = random(rng, Gamma(), shape, scale)

has_output_grad(::Gamma) = true
has_argument_grads(::Gamma) = (true, true)

export gamma
