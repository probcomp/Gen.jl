"""
    @mixturedist base_dist arg_dims output_type

Defines a new type for a mixture distribution family, and evaluates to a singleton element of that type.

The first argument `base_dist` must have type Gen.Distribution, and defines the distribution family of each component in the mixture.

The second argument `arg_dims` must be a literal tuple of Ints, of length equal to the number of arguments taken by the base distribution family.
A value of 0 at a position in the tuple an indicates that the corresponding argument to the base distribution is a scalar, and integer values of i for i >= 1 indicate that the corresponding argument is an i-dimensional array.
(Currently, only base distribution families with all scalar arguments are supported.)

The third argument, `output_type, should match the output type of the base distribution.

Example:

    mixture_of_normals = @mixturedist normal (0, 0) Float64
"""
macro mixturedist(base_dist, arg_dims, output_type)
    name = gensym(Symbol("Mixture_$(base_dist)"))
    if isa(arg_dims, Expr) && (arg_dims.head == :tuple) && all([isa(dim, Int) for dim in arg_dims.args])
        dims = Vector{Int}(arg_dims.args)
    else
        error("invalid syntax")
    end
    args = [Symbol("arg$i") for i in 1:length(dims)]

    get_arg_expr(i::Int, component) = Expr(:ref, args[i], fill(Colon(), dims[i])..., component)
    
    return quote
        struct $name <: Distribution{$output_type} end

        Gen.has_output_grad(::$(esc(name))) = has_output_grad($base_dist)
        Gen.has_argument_grads(::$(esc(name))) = (true, has_argument_grads($base_dist)...)
        Gen.is_discrete(::$(esc(name))) = is_discrete($base_dist)

        function Gen.random(::$(esc(name)), weights, $(args...))
            j = categorical(weights)
            return random($base_dist, $([get_arg_expr(i, :j) for i in 1:length(args)]...))
        end

        function Gen.logpdf(::$(esc(name)), x, weights, $(args...))
            log_densities = [Gen.logpdf($base_dist, x, $((get_arg_expr(i, :j) for i in 1:length(args))...))
                    for j in 1:length(weights)]
            log_densities = log_densities .+ log.(weights)
            return logsumexp(log_densities)
        end

        function Gen.logpdf_grad(::$(esc(name)), x, weights, $(args...))
            log_densities = [Gen.logpdf($base_dist, x, $((get_arg_expr(i, :j) for i in 1:length(args))...))
                    for j in 1:length(weights)]
            log_weighted_densities = log_densities .+ log.(weights)
            relative_weighted_densities = exp.(log_weighted_densities .- logsumexp(log_weighted_densities))

            # log_grads[j] contains the gradients for the j'th component in the mixture
            log_grads = [Gen.logpdf_grad($base_dist, x, $((get_arg_expr(i, :j) for i in 1:length(args))...))
                    for j in 1:length(weights)]

            # gradient with respect to x
            log_grads_x = [log_grad[1] for log_grad in log_grads]
            x_grad = if has_output_grad($base_dist)
                sum(log_grads_x .* relative_weighted_densities)
            else
                nothing
            end

            # gradients with respect to the weights
            weights_grad = exp.(log_densities .- logsumexp(log_weighted_densities))

            # gradients with respect to each argument
            $(Expr(:block,
                [quote
                    if has_argument_grads($base_dist)[$i]

                        $(if dims[i] == 0
                            Expr(:block, 
                                # gradients of each component log density with respect to argument i
                                :($(Symbol("$(arg)_component_grad")) = [log_grad[$(i+1)] for log_grad in log_grads]),
                                # gradient of the mixture log density respect to argument i
                                :($(Symbol("$(arg)_mixture_grad")) = $(Symbol("$(arg)_component_grad")) .* relative_weighted_densities))
                        else
                            Expr(:block,
                                # gradients of each component log density with respect to argument i
                                :($(Symbol("$(arg)_component_grad")) = cat([log_grad[$(i+1)] for log_grad in log_grads]..., dims=$(dims[i] + 1))),
                                # gradient of the mixture log density respect to argument i
                                :($(Symbol("$(arg)_mixture_grad")) = $(Symbol("$(arg)_component_grad")) .* reshape(relative_weighted_densities, $((1 for d in 1:dims[i])..., length(dims)))))
                        end)
                    else
                        $(Symbol("$(arg)_mixture_grad")) = nothing
                    end
                end for (i, arg) in enumerate(args)]...
            ))
            return $(Expr(:tuple, :(x_grad), :(weights_grad), [:($(Symbol("$(arg)_mixture_grad"))) for arg in args]...))
        end

        (::$(esc(name)))(weights, $(args...)) = random($(esc(name))(), weights, $(args...))

        $(esc(name))()
    end # end quote

end # end macro mixture

export @mixturedist
