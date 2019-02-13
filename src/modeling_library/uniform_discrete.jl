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

export uniform_discrete
