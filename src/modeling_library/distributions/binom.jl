struct Binomial <: Distribution{Int} end

"""
	binom(n::Integer, p::Real)
Sample an `Int` from the Binomial distribution with parameters `n` and `p`.
"""
const binom = Binomial()

function logpdf(::Binomial, x::Integer, n::Integer, p::Real)
	if x < 0 return -Inf end
	coefficient = loggamma(n + 1) - loggamma(n + 1 - x) - loggamma(x + 1)
	coefficient + x * log(p) + (n - x) * log(1 - p)
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
