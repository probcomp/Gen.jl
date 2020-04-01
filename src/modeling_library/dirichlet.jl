struct Dirichlet <: Distribution{Vector{Float64}} end

"""
    dirichlet(alpha::AbstractArray{U}) where {U<:Real}

Samples a `Vector{Float64}` value from a dirichlet
"""
const dirichlet = Dirichlet()

function logpdf(::Dirichlet, x::AbstractArray{T}, alpha::AbstractArray{U}) where {T <: Real, U <: Real}
    if all(alpha .> 0.)
        dist = Distributions.Dirichlet(alpha)
        Distributions.logpdf(dist, x)
    else
        -Inf
    end
end

function logpdf_grad(::Dirichlet, x::AbstractArray{T}, alpha::AbstractArray{U}) where {T <: Real, U <: Real}
    error("Not Implemented")
    (nothing, nothing)
end

function random(::Dirichlet, alpha::AbstractVector{U}) where {U <: Real}
    """ sample data from Dirichlet distribution with parameter vector alpha
    """
    rand(Distributions.Dirichlet(alpha))
end


(::Dirichlet)(alpha) = random(Dirichlet(), alpha)

has_output_grad(::Dirichlet) = false, false
# has_argument_grads(::Dirichlet) = (true, true)

export dirichlet