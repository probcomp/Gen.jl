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
struct ProductDistribution{T} <: Distribution{T}
    K::Int
    distributions::Vector{<:Distribution}
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
        type = typeof(dist)
        while supertype(type) != Any
            type = supertype(type)
        end
        push!(types, type.parameters[1])

        _has_output_grads = _has_output_grads && has_output_grad(dist)
        _is_discrete = _is_discrete && is_discrete(dist)

        grads_data = has_argument_grads(dist)
        append!(_has_argument_grads, grads_data)
        push!(_num_args, length(grads_data))
        push!(_starting_args, start_pos)
        start_pos += length(grads_data)
    end

    return ProductDistribution{Tuple{types...}}(
        length(distributions),
        collect(distributions),
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

Gen.random(dist::ProductDistribution, component_args_flat...) =
    [random(dist.distributions[k], extract_args_for_component(dist, component_args_flat, k)...) for k in 1:dist.K]

Gen.logpdf(dist::ProductDistribution, x, component_args_flat...) =
    sum(Gen.logpdf(dist.distributions[k], x[k], extract_args_for_component(dist, component_args_flat, k)...) for k in 1:dist.K)

function Gen.logpdf_grad(dist::ProductDistribution, x, component_args_flat...)
    logpdf_grads = [Gen.logpdf_grad(dist.distributions[k], x[k], extract_args_for_component(dist, component_args_flat, k)...) for k in 1:dist.K]
    x_grad = if dist.has_output_grad
        tuple((grads[1] for grads in logpdf_grads)...)
    else
        nothing
    end
    arg_grads = vcat((collect(grads[2:end]) for grads in logpdf_grads)...)
    return (x_grad, arg_grads...)
end

export ProductDistribution
