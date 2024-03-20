import LinearAlgebra

struct MultivariateNormal <: Distribution{Vector{Float64}} end

"""
    mvnormal(mu::AbstractVector{T}, cov::AbstractMatrix{U}} where {T<:Real,U<:Real}

Samples a `Vector{Float64}` value from a multivariate normal distribution.
"""
const mvnormal = MultivariateNormal()

function logpdf(::MultivariateNormal, x::AbstractVector{T}, mu::AbstractVector{U},
                cov::AbstractMatrix{V}) where {T <: Real, U <: Real, V <: Real}
    dist = Distributions.MvNormal(mu, LinearAlgebra.Symmetric(cov))
    Distributions.logpdf(dist, x)
end

function logpdf_grad(::MultivariateNormal, x::AbstractVector{T}, mu::AbstractVector{U},
                cov::AbstractMatrix{V}) where {T <: Real,U <: Real, V <: Real}
    dist = Distributions.MvNormal(mu, LinearAlgebra.Symmetric(cov))
    inv_cov = Distributions.invcov(dist)

    x_deriv = Distributions.gradlogpdf(dist, x)
    mu_deriv = -x_deriv
    cov_deriv = -0.5 * (inv_cov - (mu_deriv * transpose(mu_deriv)))

    (x_deriv, mu_deriv, cov_deriv)
end

function random(rng::AbstractRNG, ::MultivariateNormal, mu::AbstractVector{U},
                cov::AbstractMatrix{V}) where {U <: Real, V <: Real}
    rand(rng, Distributions.MvNormal(mu, LinearAlgebra.Symmetric(cov)))
end

(dist::MultivariateNormal)(mu, cov) = dist(default_rng(), mu, cov)
(::MultivariateNormal)(rng::AbstractRNG, mu, cov) = random(rng, MultivariateNormal(), mu, cov)

has_output_grad(::MultivariateNormal) = true
has_argument_grads(::MultivariateNormal) = (true, true)

export mvnormal
