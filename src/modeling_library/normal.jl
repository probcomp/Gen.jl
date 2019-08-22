struct Normal <: Distribution{Float64} end
struct BroadcastedNormal <: Distribution{Array{Float64}} end

function broadcast_shapes_or_crash(args...)
    # This expression produces a DimensionMismatch exception if the shapes are
    # incompatible.  There seems to currently be no ready-made exception-free
    # way to check this, and I think it makes more sense to wait until that
    # capability exists in Julia proper than to re-implement it here.
    s = Base.Broadcast.broadcast_shape(map(size, args)...)
end

function assert_has_shape(x, expected_shape; msg="Shape assertion failed")
    if size(x) != expected_shape
        throw(DimensionMismatch(string(msg,
                                       " Expected shape: $expected_shape",
                                       " Actual shape: $(size(x))")))
    end
    nothing
end


"""
    normal(mu::Real, std::Real)

Samples a `Float64` value from a normal distribution.
"""
const normal = Normal()

"""
    broadcasted_normal(mu::AbstractArray{<:Real, N1},
                       std::AbstractArray{<:Real, N2}) where {N1, N2}

Samples an `Array{Float64, max(N1, N2)}` of shape
`Broadcast.broadcast_shapes(size(mu), size(std))` where each element is
independently normally distributed.  This is equivalent to (a reshape of) a
multivariate normal with diagonal covariance matrix, but its implementation is
more efficient than that of the more general `mvnormal` for this case.

The shapes of `mu` and `std` must be broadcast-compatible.

If all args are 0-dimensional arrays, then sampling via
`broadcasted_normal(...)` returns a `Float64` rather than properly returning an
`Array{Float64, 0}`.  This is consistent with Julia's own inconsistency on the
matter:

```jldoctest
julia> typeof(ones())
Array{Float64,0}

julia> typeof(ones() .* ones())
Float64
```
"""
const broadcasted_normal = BroadcastedNormal()

function logpdf(::Normal, x::Real, mu::Real, std::Real)
    var = std * std
    diff = x - mu
    -(diff * diff)/ (2.0 * var) - 0.5 * log(2.0 * pi * var)
end

function logpdf(::BroadcastedNormal,
                x::Union{AbstractArray{<:Real}, Real},
                mu::Union{AbstractArray{<:Real}, Real},
                std::Union{AbstractArray{<:Real}, Real})
    assert_has_shape(x, broadcast_shapes_or_crash(mu, std);
                     msg="Shape of `x` does not agree with the sample space")
    var = std .* std
    diff = x .- mu
    sum(-(diff .* diff) ./ (2.0 * var) .- 0.5 * log.(2.0 * pi * var))
end

function logpdf_grad(::Normal, x::Real, mu::Real, std::Real)
    precision = 1. / (std * std)
    diff = mu - x
    deriv_x = diff * precision
    deriv_mu = -deriv_x
    deriv_std = -1. / std + (diff * diff) / (std * std * std)
    (deriv_x, deriv_mu, deriv_std)
end

function logpdf_grad(::BroadcastedNormal,
                     x::Union{AbstractArray{<:Real}, Real},
                     mu::Union{AbstractArray{<:Real}, Real},
                     std::Union{AbstractArray{<:Real}, Real})
    assert_has_shape(x, broadcast_shapes_or_crash(mu, std);
                     msg="Shape of `x` does not agree with the sample space")
    precision = 1.0 ./ (std .* std)
    diff = mu .- x
    deriv_x = sum(diff .* precision)
    deriv_mu = sum(-deriv_x)
    deriv_std = sum(-1.0 ./ std .+ (diff .* diff) ./ (std .* std .* std))
    (deriv_x, deriv_mu, deriv_std)
end

random(::Normal, mu::Real, std::Real) = mu + std * randn()

function random(::BroadcastedNormal,
                mu::Union{AbstractArray{<:Real}, Real},
                std::Union{AbstractArray{<:Real}, Real})
    broadcast_shape = broadcast_shapes_or_crash(mu, std)
    mu .+ std .* randn(broadcast_shape)
end

(::Normal)(mu, std) = random(Normal(), mu, std)
(::BroadcastedNormal)(mu, std) = random(BroadcastedNormal(), mu, std)

has_output_grad(::Normal) = true
has_argument_grads(::Normal) = (true, true)

has_output_grad(::BroadcastedNormal) = true
has_argument_grads(::BroadcastedNormal) = (true, true)

export normal
export broadcasted_normal
