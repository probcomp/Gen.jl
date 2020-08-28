struct Cauchy <: Distribution{Float64} end

"""
    cauchy(x0::Real, gamma::Real)

Sample a `Float64` value from a Cauchy distribution with location x0 and scale gamma.
"""
const cauchy = Cauchy()

function logpdf(::Cauchy, x::Real, x0::Real, gamma::Real)
    return Distributions.logpdf(Distributions.Cauchy(x0, gamma), x)
end

function logpdf_grad(::Cauchy, x::Real, x0::Real, gamma::Real)
    x_x0 = x - x0
    x_x0_sq = x_x0^2
    gamma_sq = gamma^2
    deriv_x0 =  2 * x_x0 / (gamma_sq + x_x0_sq)
    deriv_x = - deriv_x0
    deriv_gamma = (x_x0_sq - gamma_sq) / (gamma * (gamma_sq + x_x0_sq))
    (deriv_x, deriv_x0, deriv_gamma)
end

is_discrete(::Cauchy) = false

random(::Cauchy, x0::Real, gamma::Real) = rand(Distributions.Cauchy(x0, gamma))

(::Cauchy)(x0::Real, gamma::Real) = random(Cauchy(), x0, gamma)

has_output_grad(::Cauchy) = true
has_argument_grads(::Cauchy) = (true, true)

export cauchy
