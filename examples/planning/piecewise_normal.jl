struct PieceWiseNormal <: Gen.Distribution{Float64} end
const piecewise_normal = PieceWiseNormal()

# piece 1 has interval (-Inf, boundaries[1])
# piece 2 has interval (boundaries[1], boundaries[2])
# ..
# piece n has interval (boundaries[n-1], boundaries[n])
# piece n+1 has interval (boundaries[n], Inf)

function Gen.random(::PieceWiseNormal, probabilities::Vector{Float64},
                    mus::Vector{Float64}, stds::Vector{Float64},
                    boundaries::Vector{Float64})
    n_segments = length(probabilities)
    @assert n_segments == length(mus)
    @assert n_segments == length(stds)
    @assert n_segments == length(boundaries)+1
    idx = categorical(probabilities)
    dist = Distributions.TruncatedNormal(
        mus[idx],
        stds[idx],
        idx == 1 ? -Inf : boundaries[idx-1], idx == n_segments ? Inf : boundaries[idx])
    rand(dist)
end

function Gen.logpdf(::PieceWiseNormal, x::Float64, probabilities::Vector{Float64},
                    mus::Vector{Float64}, stds::Vector{Float64},
                    boundaries::Vector{Float64})
    n_segments = length(probabilities)
    @assert n_segments == length(mus)
    @assert n_segments == length(stds)
    @assert n_segments == length(boundaries)+1
    log_probs = Vector{Float64}(undef, n_segments)
    for idx=1:length(probabilities)
        dist = Distributions.TruncatedNormal(
            mus[idx],
            stds[idx],
            idx == 1 ? -Inf : boundaries[idx-1], idx == n_segments ? Inf : boundaries[idx])
        log_probs[idx] = log(probabilities[idx]) + Distributions.logpdf(dist, x)
    end
    logsumexp(log_probs)
end

Gen.has_output_grad(::PieceWiseNormal) = false
Gen.has_argument_grads(::PieceWiseNormal) = (false, false, false, false)
