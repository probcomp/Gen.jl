struct Laplace <: Distribution{Float64} end

"""
    laplace(loc::Real, scale::Real)

Sample a `Float64` from a laplace distribution.
"""
const laplace = Laplace()

function logpdf(::Laplace, x::Real, loc::Real, scale::Real)
    diff = abs(x - loc)
    -diff / scale - log(2.0 * scale)
end

function logpdf_grad(::Laplace, x::Real, loc::Real, scale::Real)
    precision = 1. / scale
    if x > loc
        diff = -1.
    else
        diff = 1.
    end
    deriv_x = diff * precision
    deriv_loc = -deriv_x
    deriv_scale = -1/scale + abs(x-loc) / (scale * scale)
    (deriv_x, deriv_loc, deriv_scale)
end

function random(::Laplace, loc::Real, scale::Real)
    rand(Distributions.Laplace(loc, scale))
end

is_discrete(::Laplace) = false

(::Laplace)(loc, scale) = random(Laplace(), loc, scale)

has_output_grad(::Laplace) = true
has_argument_grads(::Laplace) = (true, true)

export laplace
