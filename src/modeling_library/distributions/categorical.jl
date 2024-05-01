struct Categorical <: Distribution{Int} end

"""
    categorical(probs::AbstractArray{U, 1}) where {U <: Real}

Given a vector of probabilities `probs` where `sum(probs) = 1`, sample an `Int` `i` from the set {1, 2, .., `length(probs)`} with probability `probs[i]`.
"""
const categorical = Categorical()

function logpdf(::Categorical, x::Int, probs::AbstractArray{U,1}) where {U <: Real}
    (x > 0 && x <= length(probs)) ? log(probs[x]) : -Inf
end

function logpdf_grad(::Categorical, x::Int, probs::AbstractArray{U,1})  where {U <: Real}
    grad = zeros(length(probs))
    grad[x] = 1.0 / probs[x]
    (nothing, grad)
end

function random(rng::AbstractRNG, ::Categorical, probs::AbstractArray{U,1}) where {U <: Real}
    rand(rng, Distributions.Categorical(probs))
end
is_discrete(::Categorical) = true

(dist::Categorical)(probs) = dist(default_rng(), probs)
(::Categorical)(rng::AbstractRNG, probs) = random(rng, Categorical(), probs)

has_output_grad(::Categorical) = false
has_argument_grads(::Categorical) = (true,)

export categorical
