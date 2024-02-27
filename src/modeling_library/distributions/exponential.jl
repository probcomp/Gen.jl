struct Exponential <: Distribution{Float64} end

"""
    exponential(rate::Real)

Sample a `Float64` from the exponential distribution with rate parameter `rate`.
"""
const exponential = Exponential()

function Gen.logpdf(::Exponential, x::Real, rate::Real)
    scale = 1/rate
    Distributions.logpdf(Distributions.Exponential(scale), x)
end

function Gen.logpdf_grad(::Exponential, x::Real, rate::Real)
    scale = 1/rate
    x_grad = Distributions.gradlogpdf(Distributions.Exponential(scale), x)
    rate_grad = 1/rate - x
    (x_grad, rate_grad)
end

function Gen.random(rng::AbstractRNG, ::Exponential, rate::Real)
    scale = 1/rate
    rand(rng, Distributions.Exponential(scale))
end

is_discrete(::Exponential) = false

(dist::Exponential)(rate) = dist(default_rng(), rate)
(::Exponential)(rng::AbstractRNG, rate) = random(rng, Exponential(), rate)

Gen.has_output_grad(::Exponential) = true
Gen.has_argument_grads(::Exponential) = (true,)

export exponential
