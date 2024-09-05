struct Poisson <: Distribution{Int} end

"""
    poisson(lambda::Real)

Sample an `Int` from the Poisson distribution with rate `lambda`.
"""
const poisson = Poisson()

function logpdf(::Poisson, x::Int, lambda::Real)
    x < 0 ? -Inf : x * log(lambda) - lambda - loggamma(x+1)
end

function logpdf_grad(::Poisson, x::Int, lambda::Real)
    (nothing, x/lambda - 1)
end


function random(::Poisson, lambda::Real)
    rand(Distributions.Poisson(lambda))
end

(::Poisson)(lambda) = random(Poisson(), lambda)
is_discrete(::Poisson) = true

has_output_grad(::Poisson) = false
has_argument_grads(::Poisson) = (true,)

export poisson
