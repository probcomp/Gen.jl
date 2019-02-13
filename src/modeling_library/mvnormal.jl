struct MultivariateNormal <: Distribution{Vector{Float64}} end

"""
    mvnormal(mu::AbstractVector{T}, cov::AbstractMatrix{U}} where {T<:Real,U<:Real}

Samples a `Vector{Float64}` value from a multivariate normal distribution.
"""
const mvnormal = MultivariateNormal()

function logpdf(::MultivariateNormal, x::AbstractVector{T}, mu::AbstractVector{U},
                cov::AbstractMatrix{V}) where {T,U,V}
    dist = Distributions.MvNormal(mu, cov)
    Distributions.logpdf(dist, x)
end

function logpdf_grad(::MultivariateNormal, x::AbstractVector{T}, mu::AbstractVector{U},
                cov::AbstractMatrix{V}) where {T,U,V}
    dist = Distributions.MvNormal(mu, cov)
    x_deriv = Distributions.gradlogpdf(dist, x)
    (x_deriv, nothing, nothing)
end

function random(::MultivariateNormal, mu::AbstractVector{U},
                cov::AbstractMatrix{V}) where {T,U,V}
    rand(Distributions.MvNormal(mu, cov))
end

(::MultivariateNormal)(mu, cov) = random(MultivariateNormal(), mu, cov)

has_output_grad(::MultivariateNormal) = true
has_argument_grads(::MultivariateNormal) = (false, false)

export mvnormal
