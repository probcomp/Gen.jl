struct TransformedDistribution{T, U} <: Distribution{T}
    base :: Distribution{U}
    # How many more parameters does this distribution have
    # than the base distribution?
    nArgs :: Int8
    # forward is a U, arg... -> T function,
    # and backward is a T, arg... -> U function,
    # such that for any `args`, we have
    # backward(forward(u, args...), args...) == u
    # and
    # forward(backward(t, args...), args...) == t.
    # Note that if base is a continuous distribution, then
    # forward and backward must be differentiable.
    forward :: Function
    backward :: Function
    backward_grad :: Function
end

function random(rng::AbstractRNG, d::TransformedDistribution{T, U}, args...)::T where {T, U}
    d.forward(random(rng, d.base, args[d.nArgs+1:end]...), args[1:d.nArgs]...)
end

function logpdf_correction(d::TransformedDistribution{T, U}, x, args) where {T, U}
    log(abs(d.backward_grad(x, args...)[1]))
end

function logpdf(d::TransformedDistribution{T, U}, x::T, args...) where {T, U}
    orig_x = d.backward(x, args[1:d.nArgs]...)
    orig_logpdf = logpdf(d.base, orig_x, args[d.nArgs+1:end]...)

    if is_discrete(d.base)
        orig_logpdf
    else
        orig_logpdf + logpdf_correction(d, x, args[1:d.nArgs])
    end
end

function logpdf_grad(d::TransformedDistribution{T, U}, x::T, args...) where {T, U}
    orig_x = d.backward(x, args[1:d.nArgs]...)
    base_grad = logpdf_grad(d.base, orig_x, args[d.nArgs+1:end]...)

    if is_discrete(d.base) || !has_output_grad(d.base)
        # TODO: should this be nothing or 0?
        return (base_grad[1], fill(nothing, d.nArgs)..., base_grad[2:end]...)
    else
        transformation_grad = d.backward_grad(x, args[1:d.nArgs]...)
        correction_grad = ReverseDiff.gradient(v -> logpdf_correction(d, v[1], v[2:end]), [x, args[1:d.nArgs]...])
        # TODO: Will this sort of thing work if the arguments w.r.t. which we are taking
        # gradients are themselves vector-valued?
        full_grad = (transformation_grad .* base_grad[1]) .+ correction_grad
        return (full_grad..., base_grad[2:end]...)
    end
end

is_discrete(d::TransformedDistribution{T, U}) where {T, U} = is_discrete(d.base)

(d::TransformedDistribution)(args...) = d(default_rng(), args...)
(d::TransformedDistribution{T, U})(rng::AbstractRNG, args...) where {T, U} = random(rng, d, args...)

function has_output_grad(d::TransformedDistribution{T, U}) where {T, U}
    has_output_grad(d.base)
end

function has_argument_grads(d::TransformedDistribution{T, U}) where {T, U}
    if is_discrete(d.base) || !has_output_grad(d.base)
        (fill(false, d.nArgs)..., has_argument_grads(d.base)...)
    else
        (fill(true, d.nArgs)..., has_argument_grads(d.base)...)
    end
end
