struct Binomial <: Distribution{Int} end

"""
	binom(n::Integer, p::Real)
Sample an `Int` from the Binomial distribution with parameters `n` (number of
trials) and `p` (probability of success in each trial).
"""
const binom = Binomial()

function logpdf(::Binomial, x::Integer, n::Integer, p::Real)
    Distributions.logpdf(Distributions.Binomial(n, p), x)
end

function logpdf_grad(::Binomial, x::Integer, n::Integer, p::Real)
	(nothing, nothing, (x / p - (n - x) / (1 - p)))
end

function random(::Binomial, n::Integer, p::Real)
	rand(Distributions.Binomial(n, p))
end

(::Binomial)(n, p) = random(Binomial(), n, p)

has_output_grad(::Binomial) = false
has_argument_grads(::Binomial) = (false, true)

export binom
