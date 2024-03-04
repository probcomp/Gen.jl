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

        grads_data = has_argument_grads(dist)
        append!(_has_argument_grads, grads_data)
        push!(_num_args, length(grads_data))
        push!(_starting_args, start_pos)
        start_pos += length(grads_data)
    end

    return ProductDistribution{Tuple{types...}}(
        length(distributions),
        collect(distributions),
        all(has_output_grad(dist) for dist in distributions),
        Tuple(_has_argument_grads),
        all(is_discrete(dist) for dist in distributions),
        _num_args,
        _starting_args)
end

function Gen.random(dist::ProductDistribution, factor_args_flat...)
    factor_args = [factor_args_flat[dist.starting_args[i]:dist.starting_args[i]+dist.num_args[i]-1] for i in 1:dist.K]
    return [random(dist.distributions[i], factor_args[i]...) for i in 1:dist.K]
end

function Gen.logpdf(dist::ProductDistribution, x, factor_args_flat...)
    factor_args = [factor_args_flat[dist.starting_args[i]:dist.starting_args[i]+dist.num_args[i]-1] for i in 1:dist.K]
    return sum(Gen.logpdf(dist.distributions[i], x[i], factor_args[i]...) for i in 1:dist.K)
end

function Gen.logpdf_grad(dist::ProductDistribution, x, factor_args_flat...)
    factor_args = [factor_args_flat[dist.starting_args[i]:(dist.starting_args[i]+dist.num_args[i]-1)] for i in 1:dist.K]
    logpdf_grads = [Gen.logpdf_grad(dist.distributions[i], x[i], factor_args[i]...) for i in 1:dist.K]

    x_grad = if dist.has_output_grad
        [grads[1] for grads in logpdf_grads]
    else
        nothing
    end

    arg_grads = vcat((collect(grads[2:end]) for grads in logpdf_grads)...)

    return (x_grad, arg_grads...)
end

export ProductDistribution
