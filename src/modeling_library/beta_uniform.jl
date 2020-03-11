# TODO allow the lower and upper bounds to be changed, like uniform.

struct BetaUniformMixture <: Distribution{Float64} end

"""
    beta_uniform(theta::Real, alpha::Real, beta::Real)

Samples a `Float64` value from a mixture of a uniform distribution on [0, 1] with probability `1-theta` and a beta distribution with parameters `alpha` and `beta` with probability `theta`.
"""
const beta_uniform = BetaUniformMixture()

function logpdf(::BetaUniformMixture, x::Real, theta::Real, alpha::Real, beta::Real)
    if x < 0 || x > 1
        -Inf
    else
        lbeta = log(theta) + logpdf(Beta(), x, alpha, beta)
        luniform = log(1.0 - theta)
        logsumexp(lbeta, luniform)
    end
end

function logpdf_grad(::BetaUniformMixture, x::Real, theta::Real, alpha::Real, beta::Real)
    beta_logpdf = logpdf(Beta(), x, alpha, beta)
    uniform_logpdf = logpdf(uniform_continuous, x, 0., 1.)
    beta_grad = logpdf_grad(Beta(), x, alpha, beta)
    uniform_grad = logpdf_grad(uniform_continuous, x, 0., 1.)
    w1 = 1. / (1. + exp(log(1. - theta) + uniform_logpdf - log(theta) - beta_logpdf))
    w2 = 1. - w1
    x_deriv = w1 * beta_grad[1] + w2 * uniform_grad[1]
    alpha_deriv = w1 * beta_grad[2]
    beta_deriv = w1 * beta_grad[3]
    theta_deriv = (exp(beta_logpdf) - exp(uniform_logpdf)) / (theta * exp(beta_logpdf) + (1. - theta) * exp(uniform_logpdf))
    (x_deriv, theta_deriv, alpha_deriv, beta_deriv)
end

function random(::BetaUniformMixture, theta::Real, alpha::Real, beta::Real)
    if bernoulli(theta)
        random(Beta(), alpha, beta)
    else
        random(uniform_continuous, 0., 1.)
    end
end

(::BetaUniformMixture)(theta, alpha, beta) = random(BetaUniformMixture(), theta, alpha, beta)

is_discrete(::BetaUniformMixture) = false

has_output_grad(::BetaUniformMixture) = true
has_argument_grads(::BetaUniformMixture) = (true, true, true)

export beta_uniform
