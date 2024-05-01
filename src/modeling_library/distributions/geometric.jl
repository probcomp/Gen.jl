struct Geometric <: Distribution{Int} end

"""
    geometric(p::Real)

Sample an `Int` from the Geometric distribution with parameter `p`.
"""
const geometric = Geometric()

function Gen.logpdf(::Geometric, x::Int, p::Real)
    Distributions.logpdf(Distributions.Geometric(p), x)
end

function Gen.logpdf_grad(::Geometric, x::Int, p::Real)
    if x >= 0
        p_grad = 1/p - (1/(1-p) * x)
    else
        p_grad = 0.0
    end
    (nothing, p_grad)
end

function Gen.random(rng::AbstractRNG, ::Geometric, p::Real)
    rand(rng, Distributions.Geometric(p))
end

is_discrete(::Geometric) = true

(dist::Geometric)(p) = dist(default_rng(), p)
(::Geometric)(rng::AbstractRNG, p) = random(rng, Geometric(), p)

Gen.has_output_grad(::Geometric) = false
Gen.has_argument_grads(::Geometric) = (true,)

export geometric
