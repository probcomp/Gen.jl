##################################################################
# homogeneous mixture: arbitrary number of the same distribution #
##################################################################

"""
    @hom_mixture output_type base_dist arg_dims

Defines a new type for a mixture distribution family, and evaluates to a singleton element of that type.

The first argument, `output_type`, should match the output type of the base distribution.

The second argument `base_dist` must have type Gen.Distribution, and defines the distribution family of each component in the mixture.

The third argument `arg_dims` must be a literal tuple of Ints, of length equal to the number of arguments taken by the base distribution family.
A value of 0 at a position in the tuple an indicates that the corresponding argument to the base distribution is a scalar, and integer values of i for i >= 1 indicate that the corresponding argument is an i-dimensional array.

Example:

    mixture_of_normals = @hom_mixture Float64 normal (0, 0)
"""
macro hom_mixture(output_type, base_dist, arg_dims)
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

end # end macro hom_mixture

export @hom_mixture

##############################################################################
# heterogeneous mixture: fixed number of potentially different distributions #
##############################################################################

const mixed_syntax_err_msg ="Expected syntax of the form @mixed Float64 (normal, 2) (uniform, 2)"
const mixed_weight_vector_err_msg = "length of weight vector must match the number of components"

"""
    @het_mixture output_type (base_dist1, num_args1) (base_dist2, num_args2), ...

Defines a new type for a mixture distribution family, and evaluates to a singleton element of that type.

The first argument, `output_type`, should match the output type of all of the base distributions.

The remainder of arguments are tuples, where the first element must have type Gen.Distribution, and defines the distribution family of this component in the mixture, and the second element is a literal `Int` that gives the the number of arguments taken by this distribution family.

Example:

    uniform_beta_mixture = @het_mixture Float64 (uniform, 2) (beta, 2)
"""
macro het_mixture(output_type, base_dists_with_num_args...)
    base_dists = Any[]
    num_args = Int[]
    starting_args = Int[]
    for expr in base_dists_with_num_args
        if (!isa(expr, Expr) || expr.head != :tuple || length(expr.args) != 2 || !isa(expr.args[2], Int))
            error(mixed_syntax_err_msg)
        end
        push!(base_dists, expr.args[1])
        push!(starting_args, sum(num_args) + 1)
        push!(num_args, expr.args[2])
    end
    name = Symbol("Mixture" * reduce(*, "_$d" for d in base_dists))
    num_components = length(base_dists)

    args = Symbol[]
    for (component, (dist, num)) in enumerate(zip(base_dists, num_args))
        for cur_arg in 1:num
            push!(args, Symbol("$(dist)$(component)_arg$cur_arg"))
        end
    end

    component_arg_grads = Any[]
    for component in 1:num_components
        for i in 1:num_args[component]
            push!(component_arg_grads,
                :(relative_weighted_densities[$component] * log_grads[$component][$i+1]))
        end
    end

    return quote
        struct $name <: Distribution{$output_type}
            has_output_grad::Bool
            has_argument_grads::Tuple{$([QuoteNode(Bool) for i in 1:(1 + length(args))]...)}
            is_discrete::Bool
            base_dists::Vector{Distribution}
        end

        function $(esc(name))()
            _has_output_grad = true
            _has_argument_grads = Bool[true] # weights
            _is_discrete = true
            _base_dists = Vector{Distribution}()
            for dist in ($(base_dists...),)
                push!(_base_dists, dist)
                _has_output_grad = _has_output_grad && has_output_grad(dist)
                for has_arg_grad in has_argument_grads(dist)
                    push!(_has_argument_grads, has_arg_grad)
                end
                _is_discrete = _is_discrete && is_discrete(dist)
            end
            return $(esc(name))(_has_output_grad, tuple(_has_argument_grads...), _is_discrete, _base_dists)
        end

        # TODO implementation can probably be sped up with generated functions

        function args_for_component(args_tuple, component::Int)
            start_arg = $(QuoteNode(starting_args))[component]
            n = $(QuoteNode(num_args))[component]
            return args_tuple[start_arg:start_arg+n-1]
        end

        const singleton = $(esc(name))()

        Gen.has_output_grad(::$(esc(name))) = singleton.has_output_grad
        Gen.has_argument_grads(::$(esc(name))) = singleton.has_argument_grads
        Gen.is_discrete(::$(esc(name))) = singleton.is_discrete

        function Gen.random(::$(esc(name)), weights, $(args...))
            (length(weights) != $(QuoteNode(num_components))) && error(mixed_weight_vector_err_msg)
            component = categorical(weights)
            args_tuple = ($(args...),)
            return random(singleton.base_dists[component], args_for_component(args_tuple, component)...)
        end

        function Gen.logpdf(::$(esc(name)), x, weights, $(args...))
            (length(weights) != $(QuoteNode(num_components))) && error(mixed_weight_vector_err_msg)
            args_tuple = ($(args...),)
            log_densities = [Gen.logpdf(
                    singleton.base_dists[component], x,
                    args_for_component(args_tuple, component)...)
                for component in 1:$(QuoteNode(num_components))]
            log_densities = log_densities .+ log.(weights)
            return logsumexp(log_densities)
        end
    
        function Gen.logpdf_grad(::$(esc(name)), x, weights, $(args...))
            (length(weights) != $(QuoteNode(num_components))) && error(mixed_weight_vector_err_msg)
            args_tuple = ($(args...),)
            log_densities = [Gen.logpdf(
                    singleton.base_dists[component], x,
                    args_for_component(args_tuple, component)...)
                for component in 1:$(QuoteNode(num_components))]
            log_weighted_densities = log_densities .+ log.(weights)
            relative_weighted_densities = exp.(log_weighted_densities .- logsumexp(log_weighted_densities))

            # log_grads[component] contains the gradients for that component in the mixture
            log_grads = [Gen.logpdf_grad(
                    singleton.base_dists[component], x,
                    args_for_component(args_tuple, component)...)
                for component in 1:$(QuoteNode(num_components))]

            # gradient with respect to x
            log_grads_x = [log_grad[1] for log_grad in log_grads]
            x_grad = if has_output_grad(singleton)
                sum(log_grads_x .* relative_weighted_densities)
            else
                nothing
            end

            # gradients with respect to the weights
            weights_grad = exp.(log_densities .- logsumexp(log_weighted_densities))

            # gradients with respect to each argument
            return $(Expr(:tuple, :x_grad, :weights_grad, component_arg_grads...))
        end

        (::$(esc(name)))(weights, $(args...)) = random($(esc(name))(), weights, $(args...))

        singleton
    end # end quote

end # end macro het_mixture

export @het_mixture
