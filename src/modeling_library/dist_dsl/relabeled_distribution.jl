struct WithLabelArg{T, U} <: Distribution{T}
    base :: Distribution{U}
end

function logpdf(d::WithLabelArg{T, U}, x::T, collection, base_args...) where {T, U}
    if is_discrete(d.base)
        # Accumulate
        logprobs::Array{Float64, 1} = []
        for p in pairs(collection)
            (index, item) = (p.first, p.second)
            if item == x
                push!(logprobs, logpdf(d.base, index, base_args...))
            end
        end
        isempty(logprobs) ? -Inf : logsumexp(logprobs)
    else
        error("Cannot relabel a continuous distribution")
    end
end

function logpdf_grad(d::WithLabelArg{T, U}, x::T, collection, base_args...) where {T, U}
    base_arg_grads = Vector{Any}(nothing, length(base_args))

    for p in pairs(collection)
        (index, item) = (p.first, p.second)
        if item == x
            new_grads = logpdf_grad(d.base, index, base_args...)
            for (arg_idx, grad) in enumerate(new_grads[2:end])
                if base_arg_grads[arg_idx] === nothing
                    base_arg_grads[arg_idx] = grad
                elseif grad !== nothing
                    base_arg_grads[arg_idx] += grad
                end
            end
        end
    end
    (nothing, nothing, base_arg_grads...)
end

function random(rng::AbstractRNG, d::WithLabelArg{T, U}, collection, base_args...)::T where {T, U}
    collection[random(rng, d.base, base_args...)]
end

is_discrete(d::WithLabelArg{T, U}) where {T, U} = true

(d::WithLabelArg)(collection, base_args...) = d(default_rng(), collection, base_args...)
(d::WithLabelArg{T, U})(rng::AbstractRNG, collection, base_args...) where {T, U} = random(rng, d, collection, base_args...)

function has_output_grad(d::WithLabelArg{T, U}) where {T, U}
    false
end

has_argument_grads(d::WithLabelArg{T, U}) where {T, U} = (false, has_argument_grads(d.base)...)

struct RelabeledDistribution{T, U} <: Distribution{T}
    base :: Distribution{U}
    collection::Union{AbstractArray{T}, AbstractDict{U, T}}
end

function logpdf(d::RelabeledDistribution{T, U}, x::T, base_args...) where {T, U}
    if is_discrete(d.base)
        # Accumulate
        logprobs::Array{Float64, 1} = []
        for p in pairs(d.collection)
            (index, item) = (p.first, p.second)
            if item == x
                push!(logprobs, logpdf(d.base, index, base_args...))
            end
        end
        isempty(logprobs) ? -Inf : logsumexp(logprobs)
    else
        error("Cannot relabel a continuous distribution")
    end
end

function logpdf_grad(d::RelabeledDistribution{T, U}, x::T, base_args...) where {T, U}
    base_arg_grads = Vector{Any}(nothing, length(base_args))

    for p in pairs(d.collection)
        (index, item) = (p.first, p.second)
        if item == x
            new_grads = logpdf_grad(d.base, index, base_args...)
            for (arg_idx, grad) in enumerate(new_grads[2:end])
                if base_arg_grads[arg_idx] === nothing
                    base_arg_grads[arg_idx] = grad
                elseif grad !== nothing
                    base_arg_grads[arg_idx] += grad
                end
            end
        end
    end
    (nothing, base_arg_grads...)
end

function random(rng::AbstractRNG, d::RelabeledDistribution{T, U}, base_args...)::T where {T, U}
    d.collection[random(rng, d.base, base_args...)]
end

is_discrete(d::RelabeledDistribution{T, U}) where {T, U} = true

(d::RelabeledDistribution)(base_args...) = d(default_rng(), base_args...)
(d::RelabeledDistribution{T, U})(rng::AbstractRNG, base_args...) where {T, U} = random(rng, d, base_args...)

function has_output_grad(d::RelabeledDistribution{T, U}) where {T, U}
    false
end

has_argument_grads(d::RelabeledDistribution{T, U}) where {T, U} = has_argument_grads(d.base)
