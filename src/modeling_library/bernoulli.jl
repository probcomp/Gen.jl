struct Bernoulli <: Distribution{Bool} end

"""
    bernoulli(prob_true::Real)

Samples a `Bool` value which is true with given probability
"""
const bernoulli = Bernoulli()

function logpdf(::Bernoulli, x::Bool, prob::Real)
    x ? log(prob) : log(1. - prob)
end

function logpdf_grad(::Bernoulli, x::Bool, prob::Real)
    prob_grad = x ? 1. / prob : -1. / (1-prob)
    (nothing, prob_grad)
end

random(::Bernoulli, prob::Real) = rand() < prob

(::Bernoulli)(prob) = random(Bernoulli(), prob)

has_output_grad(::Bernoulli) = false
has_argument_grads(::Bernoulli) = (true,)

export bernoulli
