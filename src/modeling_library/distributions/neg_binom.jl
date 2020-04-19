struct NegativeBinomial <: Distribution{Int} end

"""
    neg_binom(r::Real, p::Real)
Sample an `Int` from a Negative Binomial distribution. Returns the number of
failures before the `r`th success in a sequence of independent Bernoulli trials.
`r` is the number of successes and `p` is the probability of success per trial.
"""
const neg_binom = NegativeBinomial()

function Gen.logpdf(::NegativeBinomial, x::Int, r::Real, p::Real)
    Gen.Distributions.logpdf(Gen.Distributions.NegativeBinomial(r, p), x)
end

function Gen.logpdf_grad(::NegativeBinomial, x::Int, r::Real, p::Real)
    r_grad = x == 0 ? log(p) : sum(1/(x+r-i) for i in 1:x) + log(p)
    p_grad = x >= 0 ? r/p - (1/(1-p) * x) : 0.0
    (nothing, r_grad, p_grad)
end

function Gen.random(::NegativeBinomial, r::Real, p::Real)
    rand(Gen.Distributions.NegativeBinomial(r, p))
end

is_discrete(::NegativeBinomial) = true

(::NegativeBinomial)(r, p) = random(NegativeBinomial(), r, p)

Gen.has_output_grad(::NegativeBinomial) = false
Gen.has_argument_grads(::NegativeBinomial) = (true, true)

export neg_binom
