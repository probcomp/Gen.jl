
##################################################################
# homogeneous mixture: arbitrary number of the same distribution #
##################################################################

"""
    @hom_mixture output_type base_dist arg_dims

Defines a new type for a mixture distribution family, and evaluates to a
singleton element of that type.

The first argument, `output_type`, should match the output type of the base
distribution.

The second argument `base_dist` must have type Gen.Distribution, and defines
the distribution family of each component in the mixture.

The third argument `arg_dims` must be a literal tuple of Ints, of length equal
to the number of arguments taken by the base distribution family.  A value of 0
at a position in the tuple an indicates that the corresponding argument to the
base distribution is a scalar, and integer values of i for i >= 1 indicate that
the corresponding argument is an i-dimensional array.

Example:

    mixture_of_normals = @hom_mixture Float64 normal (0, 0)
"""
macro hom_mixture(output_type, base_dist, arg_dims)
    name = gensym(Symbol("Mixture_$(base_dist)"))

    # validate arguments
    if isa(arg_dims, Expr) && (arg_dims.head == :tuple) && all([isa(dim, Int) for dim in arg_dims.args])
        dims = Vector{Int}(arg_dims.args)
    else
        error("Invalid arg_dims specification.")
    end
    args = [Symbol("arg$i") for i in 1:length(dims)]
    num_args = length(args)

    # expression for the i'th argument to the k'th component distribution
    arg_expr(i::Int, k_expr) = Expr(:ref, args[i], fill(Colon(), dims[i])..., k_expr)

    # list of expressions for all arguments to k'th component distribution
    arg_exprs(k_expr) = [Expr(:ref, args[i], fill(Colon(), dims[i])..., k_expr) for i in 1:num_args]
    
    return quote
        struct $name <: Distribution{$output_type} end

        Gen.has_output_grad(::$(esc(name))) = has_output_grad($base_dist)
        Gen.has_argument_grads(::$(esc(name))) = (true, has_argument_grads($base_dist)...)
        Gen.is_discrete(::$(esc(name))) = is_discrete($base_dist)

        function Gen.random(::$(esc(name)), weights, $(args...))
            K = length(weights)
            k = categorical(weights)
            return random($base_dist, $(arg_exprs(:k)...))
        end

        function Gen.logpdf(::$(esc(name)), x, weights, $(args...))
            K = length(weights)
            log_densities = [Gen.logpdf($base_dist, x, $(arg_exprs(:k)...)) for k in 1:K]
            log_densities = log_densities .+ log.(weights)
            return logsumexp(log_densities)
        end

        function Gen.logpdf_grad(::$(esc(name)), x, weights, $(args...))
            K = length(weights)
            log_densities = [Gen.logpdf($base_dist, x, $(arg_exprs(:k)...)) for k in 1:K]
            log_weighted_densities = log_densities .+ log.(weights)
            relative_weighted_densities = exp.(log_weighted_densities .- logsumexp(log_weighted_densities))

            # log_grads[k] contains the gradients for the k'th component
            log_grads = [Gen.logpdf_grad($base_dist, x, $(arg_exprs(:k)...)) for k in 1:K]

            # compute gradient with respect to x
            log_grads_x = [log_grad[1] for log_grad in log_grads]
            x_grad = if has_output_grad($base_dist)
                sum(log_grads_x .* relative_weighted_densities)
            else
                nothing
            end

            # compute gradients with respect to the weights
            weights_grad = exp.(log_densities .- logsumexp(log_weighted_densities))

            # compute gradients with respect to each argument
            $(Expr(:block, [quote
                if has_argument_grads($base_dist)[$i]
                    $(if dims[i] == 0
                        # scalar argument
                        quote
                            grads = [log_grad[$(i+1)] for log_grad in log_grads]
                            grad_weights = relative_weighted_densities
                        end
                    else
                        # array argument
                        quote
                            grads = cat([log_grad[$(i+1)] for log_grad in log_grads]..., dims=$(dims[i] + 1))
                            grad_weights = reshape(relative_weighted_densities, $((1 for d in 1:dims[i])..., length(dims)))
                        end
                    end)
                    $(Symbol("$(arg)_grad")) = grads .* grad_weights
                else
                    $(Symbol("$(arg)_grad")) = nothing
                end
            end for (i, arg) in enumerate(args)]...))

            return $(Expr(:tuple,
                :x_grad, :weights_grad,
                [:($(Symbol("$(arg)_grad"))) for arg in args]...))
        end

        # define callable behavior to be sampling
        (::$(esc(name)))(weights, $(args...)) = random($(esc(name))(), weights, $(args...))

        # evaluate to a singleton
        $(esc(name))()

    end # quote

end # macro hom_mixture

export @hom_mixture


##############################################################################
# heterogeneous mixture: fixed number of potentially different distributions #
##############################################################################

const MIXED_SYNTAX_ERR_MSG ="Expected syntax of the form @mixed Float64 (normal, 2) (uniform, 2)"
const MIXED_WEIGHT_VECTOR_ERR_MSG = "length of weight vector must match the number of components"

"""
    @het_mixture output_type (base_dist1, num_args1) (base_dist2, num_args2), ...

Defines a new type for a mixture distribution family, and evaluates to a
singleton element of that type.

The first argument, `output_type`, should match the output type of all of the
base distributions.

The remainder of arguments are tuples, where the first element must have type
Gen.Distribution, and defines the distribution family of this component in the
mixture, and the second element is a literal `Int` that gives the the number of
arguments taken by this distribution family.

Example:

    uniform_beta_mixture = @het_mixture Float64 (uniform, 2) (beta, 2)
"""
macro het_mixture(output_type, base_dists_with_num_args...)

    # parse arguments
    base_dists = Any[]
    num_args = Int[]
    starting_args = Int[]
    for expr in base_dists_with_num_args
        if (!isa(expr, Expr) || expr.head != :tuple || length(expr.args) != 2 || !isa(expr.args[2], Int))
            error(MIXED_SYNTAX_ERR_MSG)
        end
        push!(base_dists, expr.args[1])
        push!(starting_args, sum(num_args) + 1)
        push!(num_args, expr.args[2])
    end
    name = Symbol("Mixture" * reduce(*, "_$d" for d in base_dists))
    K = length(base_dists)

    component_args_flat = Symbol[]
    for (k, (dist, num)) in enumerate(zip(base_dists, num_args))
        for cur_arg in 1:num
            push!(component_args_flat, Symbol("$(dist)$(k)_arg$cur_arg"))
        end
    end

    k_arg_grads = Any[]
    for k in 1:K
        for i in 1:num_args[k]
            push!(k_arg_grads,
                :(relative_weighted_densities[$k] * log_grads[$k][$i+1]))
        end
    end

    return quote

        struct $name <: Distribution{$output_type}
            has_output_grad::Bool
            has_argument_grads::Tuple{$(fill(QuoteNode(Bool), 1+length(component_args_flat))...)}
            is_discrete::Bool
            base_dists::Vector{Distribution}
        end

        # constructor that precomputes the static attributes of the distribution family
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

        function extract_args_for_component(args_tuple, k::Int)
            start_arg = $(QuoteNode(starting_args))[k]
            n = $(QuoteNode(num_args))[k]
            return args_tuple[start_arg:start_arg+n-1]
        end

        const singleton = $(esc(name))()

        Gen.has_output_grad(::$(esc(name))) = singleton.has_output_grad
        Gen.has_argument_grads(::$(esc(name))) = singleton.has_argument_grads
        Gen.is_discrete(::$(esc(name))) = singleton.is_discrete

        function Gen.random(::$(esc(name)), weights, $(component_args_flat...))
            (length(weights) != $(QuoteNode(K))) && error(MIXED_WEIGHT_VECTOR_ERR_MSG)
            k = categorical(weights)
            args_tuple = ($(component_args_flat...),)
            return random(singleton.base_dists[k], extract_args_for_component(args_tuple, k)...)
        end

        function Gen.logpdf(::$(esc(name)), x, weights, $(component_args_flat...))
            (length(weights) != $(QuoteNode(K))) && error(MIXED_WEIGHT_VECTOR_ERR_MSG)
            args_tuple = ($(component_args_flat...),)
            log_densities = [Gen.logpdf(
                    singleton.base_dists[k], x,
                    extract_args_for_component(args_tuple, k)...)
                for k in 1:$(QuoteNode(K))]
            log_densities = log_densities .+ log.(weights)
            return logsumexp(log_densities)
        end
    
        function Gen.logpdf_grad(::$(esc(name)), x, weights, $(component_args_flat...))
            (length(weights) != $(QuoteNode(K))) && error(MIXED_WEIGHT_VECTOR_ERR_MSG)
            args_tuple = ($(component_args_flat...),)
            log_densities = [Gen.logpdf(
                    singleton.base_dists[k], x,
                    extract_args_for_component(args_tuple, k)...)
                for k in 1:$(QuoteNode(K))]
            log_weighted_densities = log_densities .+ log.(weights)
            relative_weighted_densities = exp.(log_weighted_densities .- logsumexp(log_weighted_densities))

            # log_grads[k] contains the gradients for that k in the mixture
            log_grads = [Gen.logpdf_grad(
                    singleton.base_dists[k], x,
                    extract_args_for_component(args_tuple, k)...)
                for k in 1:$(QuoteNode(K))]

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
            return $(Expr(:tuple, :x_grad, :weights_grad, k_arg_grads...))
        end

        # define callable behavior to be sampling
        (::$(esc(name)))(weights, $(component_args_flat...)) = random(
            $(esc(name))(), weights, $(component_args_flat...))

        # evaluates to the singleton of the new type
        singleton

    end # quote

end # macro het_mixture

export @het_mixture
