########################################################################
# ProductDistribution: product of fixed distributions of similar types #
########################################################################

"""
ProductDistribution(distributions::Vararg{<:Distribution})

Define new distribution that is the product of the given nonempty list of distributions having a common type.

The arguments comprise the list of base distributions.

Example:
```julia
normal_strip = ProductDistribution(uniform, normal)
```

The resulting product distribution takes `n` arguments, where `n` is the sum of the numbers of arguments taken by each distribution in the list.
These arguments are the arguments to each component distribution, in the order in which the distributions are passed to the constructor.

Example:
```julia
@gen function unit_strip_and_near_seven()
    x ~ flip_and_number(0.0, 0.1, 7.0, 0.01)
end
```
"""
struct ProductDistribution{T, Ds} <: Distribution{T}
    K::Int
    distributions::Ds
    has_output_grad::Bool
    has_argument_grads::Tuple
    is_discrete::Bool
    num_args::Vector{Int}
    starting_args::Vector{Int}
end

(dist::ProductDistribution)(args...) = random(dist, args...)

Gen.has_output_grad(dist::ProductDistribution) = dist.has_output_grad
Gen.has_argument_grads(dist::ProductDistribution) = dist.has_argument_grads
Gen.is_discrete(dist::ProductDistribution) = dist.is_discrete

function ProductDistribution(distributions::Vararg{<:Distribution})
    _has_output_grads = true
    _is_discrete = true

    types = Type[]

    _has_argument_grads = Bool[]
    _num_args = Int[]
    _starting_args = Int[]
    start_pos = 1

    for dist in distributions
        push!(types, get_return_type(dist))

        _has_output_grads = _has_output_grads && has_output_grad(dist)
        _is_discrete = _is_discrete && is_discrete(dist)

        grads_data = has_argument_grads(dist)
        append!(_has_argument_grads, grads_data)
        push!(_num_args, length(grads_data))
        push!(_starting_args, start_pos)
        start_pos += length(grads_data)
    end

    return ProductDistribution{Tuple{types...}, typeof(distributions)}(
        length(distributions),
        distributions,
        _has_output_grads,
        Tuple(_has_argument_grads),
        _is_discrete,
        _num_args,
        _starting_args)
end

function extract_args_for_component(dist::ProductDistribution, component_args_flat, k::Int)
    start_arg = dist.starting_args[k]
    n = dist.num_args[k]
    return component_args_flat[start_arg:start_arg+n-1]
end

Gen.random(dist::ProductDistribution, args...) =
    Tuple(random(d, extract_args_for_component(dist, args, k)...) for (k, d) in enumerate(dist.distributions))

Gen.logpdf(dist::ProductDistribution, x, args...) =
    sum(Gen.logpdf(d, x[k], extract_args_for_component(dist, args, k)...) for (k, d) in enumerate(dist.distributions))

function Gen.logpdf_grad(dist::ProductDistribution, x, args...)
    x_grad = ()
    arg_grads = ()
    for (k, d) in enumerate(dist.distributions)
        grads = Gen.logpdf_grad(d, x[k], extract_args_for_component(dist, args, k)...)
        x_grad = (x_grad..., grads[1])
        arg_grads = (arg_grads..., grads[2:end]...)
    end
    x_grad = dist.has_output_grad ? x_grad : nothing
    return (x_grad, arg_grads...)
end

export ProductDistribution
