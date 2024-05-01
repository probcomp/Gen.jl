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

function random(rng::AbstractRNG, ::UniformContinuous, low::Real, high::Real)
    rand(rng) * (high - low) + low
end

(dist::UniformContinuous)(low, high) = dist(default_rng(), low, high)
(::UniformContinuous)(rng::AbstractRNG, low, high) = random(rng, UniformContinuous(), low, high)

is_discrete(::UniformContinuous) = false

has_output_grad(::UniformContinuous) = true
has_argument_grads(::UniformContinuous) = (true, true)

export uniform_continuous, uniform
