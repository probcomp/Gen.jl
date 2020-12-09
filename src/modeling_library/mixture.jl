# TODO handle other distribution types

macro mixturedist(base_dist, arg_dims, output_type)
    name = gensym(Symbol("Mixture_$(base_dist)"))
    if isa(arg_dims, Expr) && (arg_dims.head == :tuple) && all([isa(dim, Int) for dim in arg_dims.args])
        dims = Vector{Int}(arg_dims.args)
        println("got dims: $dims")
        for dim in dims
            if dim != 0
                # TODO simply use arg[:,:,i] instead of args[i] below to handle distributions with array arguments
                error("only distributions with scalar arguments are currently supported")
            end
        end
    else
        error("invalid syntax")
    end
    args = [Symbol("arg$i") for i in 1:length(dims)]
    
    return quote
        struct $name <: Distribution{$output_type} end

        function Gen.random(::$(esc(name)), weights, $(args...))
            j = categorical(weights)
            return random($base_dist, $([:($(arg)[j]) for arg in args]...))
        end

        function Gen.logpdf(::$(esc(name)), x, weights, $(args...))
            log_densities = [Gen.logpdf($base_dist, x, $((:($(args[i])[j]) for i in 1:length(dims))...))
                    for j in 1:length(weights)]
            log_densities = log_densities .+ log.(weights)
            return logsumexp(log_densities)
        end

        function Gen.logpdf_grad(::$(esc(name)), x, weights, $(args...))
            log_densities = [Gen.logpdf($base_dist, x, $((:($(args[i])[j]) for i in 1:length(dims))...))
                    for j in 1:length(weights)]
            log_weighted_densities = log_densities .+ log.(weights)
            relative_weighted_densities = exp.(log_weighted_densities .- logsumexp(log_weighted_densities))

            # log_grads[j] contains the gradients for the j'th component in the mixture
            log_grads = [Gen.logpdf_grad($base_dist, x, $((:($(args[i])[j]) for i in 1:length(dims))...))
                    for j in 1:length(weights)]

            # gradient with respect to x
            log_grads_x = [log_grad[1] for log_grad in log_grads]
            x_grad = $(if output_type == :Float64 # TODO use has_output_grad instead
                :(sum(log_grads_x .* relative_weighted_densities))
            else
                nothing
            end)

            # gradients with respect to the weights
            weights_grad = exp.(log_densities .- logsumexp(log_weighted_densities))

            # gradients with respect to each argument
            $(Expr(:block,
                [quote
                    # TODO check if it actually has gradients for its arguments..
                    $(Symbol("log_grads_$(arg)")) = [log_grad[$(i+1)] for log_grad in log_grads] # gradients with respect to arg i, for each component in the mixture
                    $(Symbol("$(arg)_grad")) = $(Symbol("log_grads_$(arg)")) .* relative_weighted_densities
                end for (i, arg) in enumerate(args)]...
            ))
            return $(Expr(:tuple, :(x_grad), :(weights_grad), [:($(Symbol("$(arg)_grad"))) for arg in args]...))
        end

        (::$(esc(name)))(weights, $(args...)) = random($(esc(name))(), weights, $(args...))

        $(esc(name))()
    end # end quote

end # end macro mixture

export @mixturedist
