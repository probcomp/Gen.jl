struct Cauchy <: Distribution{Float64} end

"""
    Cauchy(x0::Real, gamma::Real)

Sample a `Float64` value from a Cauchy distribution with location x0 and scale gamma.
"""
const cauchy = Cauchy()

function logpdf(::Cauchy, x::Real, x0::Real, gamma::Real)
    return Distributions.logpdf(Distributions.Cauchy(x0, gamma), x)
end

function logpdf_grad(::Cauchy, x::Real, x0::Real, gamma::Real)
    deriv_x =  - 2 * (x - x0) / (gamma^2 + (x - x0)^2)
    deriv_x0 = 2 * (x - x0) / (gamma^2 + (x - x0)^2)
    deriv_gamma = ((x - x0)^2 - gamma^2) / (gamma * (gamma^2 + (x - x0)^2)) 
    (deriv_x, deriv_x0, deriv_gamma)
end

is_discrete(::Cauchy) = false

random(::Cauchy, x0::Real, gamma::Real) = rand(Distributions.Cauchy(x0, gamma))

(::Cauchy)(x0::Real, gamma::Real) = random(Cauchy(), x0, gamma)

has_output_grad(::Cauchy) = true
has_argument_grads(::Cauchy) = (true, true)

export cauchy