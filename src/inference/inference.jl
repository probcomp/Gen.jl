import Random

function logsumexp(arr::AbstractArray{T}) where {T <: Real}
    max_arr = maximum(arr)
    max_arr + log(sum(exp.(arr .- max_arr)))
end

function logsumexp(x1::Real, x2::Real)
    max_arr = max(x1, x2)
    max_arr + log(exp(x1 - max_arr) + exp(x2 - max_arr))
end

export logsumexp

include("mh.jl")
include("hmc.jl")
include("mala.jl")
include("importance.jl")
include("particle_filter.jl")
include("map_optimize.jl")
include("train.jl")
include("variational.jl")
