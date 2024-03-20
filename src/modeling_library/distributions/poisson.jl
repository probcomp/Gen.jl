struct Poisson <: Distribution{Int} end

"""
    poisson(lambda::Real)

Sample an `Int` from the Poisson distribution with rate `lambda`.
"""
const poisson = Poisson()

function logpdf(::Poisson, x::Int, lambda::Real)
    x < 0 ? -Inf : x * log(lambda) - lambda - loggamma(x+1)
end

function logpdf_grad(::Poisson, x::Int, lambda::Real)
    (nothing, x/lambda - 1)
end


function random(rng::AbstractRNG, ::Poisson, lambda::Real)
    rand(rng, Distributions.Poisson(lambda))
end

(dist::Poisson)(lambda) = dist(default_rng(), lambda)
(::Poisson)(rng::AbstractRNG, lambda) = random(rng, Poisson(), lambda)

is_discrete(::Poisson) = true

has_output_grad(::Poisson) = false
has_argument_grads(::Poisson) = (false,)

export poisson
