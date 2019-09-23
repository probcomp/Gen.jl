struct Binomial <: Distribution{Int} end

"""
	binomial(n::Int, p::Real)

Sample an `Int` from the Binomial distribution with parameters `n` and `p`.
"""
const binomial = Binomial()

function logpdf(::Binomial, x::Integer, n::Integer, p::Real)
	coefficient = lgamma(n + 1) - lgamma(n + 1 - x) - lgamma(x + 1)
	coefficient + x * log(p) + (n - x) * log(1 - p)
end

function logpdf_grad(::Binomial, x::Integer, n::Integer, p::Real)
	error("Not implemented")
	(nothing, nothing)
end

function random(::Binomial, n::Integer, p::Real)
	rand(Distributions.Binomial(n, p))
end

(::Binomial)(n, p) = random(Binomial(), n, p)

has_output_grad(::Binomial) = false
has_argument_grads(::Binomial) = (false,)

export binomial