struct PiecewiseUniform <: Distribution{Float64} end

"""
    piecewise_uniform(bounds, probs)

Samples a `Float64` value from a piecewise uniform continuous distribution.

There are `n` bins where `n = length(probs)` and `n + 1 = length(bounds)`.
Bounds must satisfy `bounds[i] < bounds[i+1]` for all `i`.
The probability density at `x` is zero if `x <= bounds[1]` or `x >= bounds[end]` and is otherwise `probs[bin] / (bounds[bin] - bounds[bin+1])` where `bounds[bin] < x <= bounds[bin+1]`.
"""
const piecewise_uniform = PiecewiseUniform()

function check_dims(::PiecewiseUniform, bounds, probs)
    if length(bounds) != length(probs) + 1
        error("Dimension mismatch")
    end
end

function get_bin(bounds, x)
    @assert x <= bounds[end]
    bin = 1
    while x > bounds[bin+1]
        bin += 1
    end
    @assert x > bounds[bin] && x <= bounds[bin+1]
    bin
end

function logpdf(::PiecewiseUniform, x::Real, bounds::AbstractVector{T},
                    probs::AbstractVector{U}) where {T <: Real, U <: Real}
    check_dims(piecewise_uniform, bounds, probs)

    # bounds[1]      bounds[2]           bounds[3]      bounds[4]
    # ^              ^                   ^              ^
    # |    probs[1]  |  probs[2]         | probs[3]     |
    if x <= bounds[1] || x >= bounds[end]
        -Inf
    else
        bin = get_bin(bounds, x)
        log(probs[bin]) - log(bounds[bin+1] - bounds[bin])
    end
end

function random(::PiecewiseUniform, bounds::Vector{T},
                    probs::Vector{U}) where {T <: Real, U <: Real}
    bin = categorical(probs)
    uniform_continuous(bounds[bin], bounds[bin+1])
end

(::PiecewiseUniform)(bounds, probs) = random(PiecewiseUniform(), bounds, probs)

function logpdf_grad(::PiecewiseUniform, x::Real, bounds, probs)
    check_dims(piecewise_uniform, bounds, probs)
    if x <= bounds[1] || x >= bounds[end]
        error("Out of bounds")
    end
    bin = get_bin(bounds, x)
    bounds_grad = fill(0., length(bounds))
    bin_length = bounds[bin+1] - bounds[bin]
    bounds_grad[bin] = 1. / bin_length
    bounds_grad[bin+1] = - 1. / bin_length
    probs_grad = fill(0., length(probs))
    probs_grad[bin] = 1. / probs[bin]
    (0., bounds_grad, probs_grad)
end

is_discrete(::PiecewiseUniform) = false

has_output_grad(::PiecewiseUniform) = true
has_argument_grads(::PiecewiseUniform) = (true, true)

export piecewise_uniform
