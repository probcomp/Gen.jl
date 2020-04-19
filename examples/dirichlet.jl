using Gen

#############
# dirichlet #
#############

import Distributions
struct Dirichlet <: Distribution{Vector{Float64}} end

"""
    dirichlet(alpha::AbstractVector{T}) where {T<:Real}
Sample a `Vector{Float64}` from the Dirichlet distribution with parameter vector `alpha`.
"""
const dirichlet = Dirichlet()

function Gen.logpdf(::Dirichlet, x::AbstractVector{T}, alpha::AbstractVector{U}) where {T, U}
    Distributions.logpdf(Distributions.Dirichlet(alpha), x)
end

function Gen.logpdf_grad(::Dirichlet, x::AbstractVector{T}, alpha::AbstractVector{U}) where {T, U}
    (Distributions.gradlogpdf(Distributions.Dirichlet(alpha), x), nothing, nothing)
end

function Gen.random(::Dirichlet, alpha::AbstractVector{T}) where {T}
    rand(Distributions.Dirichlet(alpha))
end

(::Dirichlet)(alpha) = random(Dirichlet(), alpha)

Gen.has_output_grad(::Dirichlet) = true
Gen.has_argument_grads(::Dirichlet) = (false,)


