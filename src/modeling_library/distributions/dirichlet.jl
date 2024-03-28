struct Dirichlet <: Distribution{Vector{Float64}} end

"""
    Dirichlet(alpha::Vector{Float64})

Sample a simplex Vector{Float64} from a Dirichlet distribution.
"""
const dirichlet = Dirichlet()

function logpdf(::Dirichlet, x::AbstractVector{T}, alpha::AbstractVector{U}) where {T <: Real, U <: Real}
    if length(x) == length(alpha) && isapprox(sum(x), 1) && all(x .>= 0) && all(alpha .>= 0)
        ll = sum((a_i - 1) * log(x_i) for (a_i, x_i) in zip(alpha, x))
        ll -= sum(loggamma.(alpha)) - loggamma(sum(alpha))
        ll
    else
        -Inf
    end
end

function logpdf_grad(::Dirichlet, x::AbstractVector{T}, alpha::AbstractVector{U}) where {T <: Real, U <: Real}
    if length(x) == length(alpha) && isapprox(sum(x), 1) && all(x .>= 0) && all(alpha .>= 0)
        deriv_x = (alpha .- 1) ./ x
        deriv_alpha = log.(x) .- digamma.(alpha) .+ digamma(sum(alpha))
        (deriv_x, deriv_alpha)
    else
        (zero(x), zero(alpha))
    end
end

function random(::Dirichlet, alpha::AbstractVector{T}) where {T <: Real}
    rand(Distributions.Dirichlet(alpha))
end

is_discrete(::Dirichlet) = false

(::Dirichlet)(alpha) = random(Dirichlet(), alpha)

has_output_grad(::Dirichlet) = true
has_argument_grads(::Dirichlet) = (true,)

export dirichlet

