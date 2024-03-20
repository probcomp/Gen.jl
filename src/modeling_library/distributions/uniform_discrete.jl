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

function random(rng::AbstractRNG, ::UniformDiscrete, low::Integer, high::Integer)
    rand(rng, Distributions.DiscreteUniform(low, high))
end
is_discrete(::UniformDiscrete) = true

(dist::UniformDiscrete)(low, high) = dist(default_rng(), low, high)
(::UniformDiscrete)(rng::AbstractRNG, low, high) = random(rng, UniformDiscrete(), low, high)

has_output_grad(::UniformDiscrete) = false
has_argument_grads(::UniformDiscrete) = (false, false)

export uniform_discrete
