# TODO allow the lower and upper bounds to be parameterized.
# use default values for the lower and upper bounds ? (0 and 1)

struct Beta <: Distribution{Float64} end

"""
    beta(alpha::Real, beta::Real)

Sample a `Float64` from a beta distribution.
"""
const beta = Beta()

function logpdf(::Beta, x::Real, alpha::Real, beta::Real)
    (x < 0 || x > 1 ? -Inf :
        ((x == 0 && alpha <= 1) ?      #Special case to avoid infinite PDF at 0
            (alpha - 1) * log(eps(typeof(x))) -
            alpha + 1 + #as if we were using x=(eps / e); even smaller than eps
            (beta - 1) * log1p(-eps(typeof(x))) - logbeta(alpha, beta) :
        ((x == 1 && beta <= 1) ?       #Special case to avoid infinite PDF at 1
            (alpha - 1) * log1p(-eps(typeof(x))) -
            beta + 1 + #as if we were using 1-x=(eps / e)
            (beta - 1) * log(eps(typeof(x))) - logbeta(alpha, beta) :
    (alpha - 1) * log(x) + (beta - 1) * log1p(-x) - logbeta(alpha, beta) )))
end

function logpdf_grad(::Beta, x::Real, alpha::Real, beta::Real)
    if 0 <= x <= 1
        deriv_x = (alpha - 1) / x - (beta - 1) / (1 - x)
        deriv_alpha = log(x) - (digamma(alpha) - digamma(alpha + beta))
        deriv_beta = log1p(-x) - (digamma(beta) - digamma(alpha + beta))
    else
        error("x is outside of support: $x")
    end
    (deriv_x, deriv_alpha, deriv_beta)
end

function random(rng::AbstractRNG, ::Beta, alpha::Real, beta::Real)
    rand(rng, Distributions.Beta(alpha, beta))
end
is_discrete(::Beta) = false

(dist::Beta)(alpha, beta) = dist(default_rng(), alpha, beta)
(::Beta)(rng::AbstractRNG, alpha, beta) = random(rng, Beta(), alpha, beta)

has_output_grad(::Beta) = true
has_argument_grads(::Beta) = (true, true)

export beta
