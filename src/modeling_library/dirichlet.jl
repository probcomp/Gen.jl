struct Dirichlet <: Distribution{Vector{Float64}} end

"""
    dirichlet(alpha::AbstractArray{U}) where {U<:Real}

Samples a `Vector{Float64}` value from a dirichlet
"""
const dirichlet = Dirichlet()

function logpdf(::Dirichlet, x::AbstractArray{T}, alpha::AbstractArray{U}) where {T <: Real, U <: Real}
    if !(isapprox(sum(x),1) & all(x .>= 0) & all(x .<= 1))
        -Inf
    elseif all(alpha .> 0.)
        sum((alpha.-1).*log.(x)) - (sum(loggamma.(alpha))-loggamma(sum(alpha)))
    else
        -Inf
    end
end

function logpdf_grad(::Dirichlet, x::AbstractArray{T}, alpha::AbstractArray{U}) where {T <: Real, U <: Real}
    if (isapprox(sum(x),1) & all(x .>= 0) & all(x .<= 1))
        println()
        deriv_x = sum((alpha.-1) ./ x)
        deriv_alpha = log.(x) .+ digamma(sum(alpha)) .- digamma.(alpha)
    else
        error("x has to be a simplex: $x")
    end
    Tuple(vcat([deriv_x, deriv_alpha]...))
end

function random(::Dirichlet, alpha::AbstractVector{U}) where {U <: Real}
    """ sample data from Dirichlet distribution with parameter vector alpha
    """
    rand(Distributions.Dirichlet(alpha))
end

is_discrete(::Dirichlet) = false
(::Dirichlet)(alpha) = random(Dirichlet(), alpha)

has_output_grad(::Dirichlet) = false, false
# has_argument_grads(::Dirichlet) = (true, true)

export dirichlet