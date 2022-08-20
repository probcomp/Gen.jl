###############################
# Core Distribution Interface #
###############################

struct DistributionTrace{T, Dist} <: Trace
    val::T
    args
    score::Float64
    dist::Dist
end
@inline dist(tr::DistributionTrace{T, Dist}) where {T, Dist} = tr.dist

abstract type Distribution{T} <: GenerativeFunction{T, DistributionTrace{T}} end
DistributionTrace{T, Dist}(val::T, args::Tuple, dist::Dist) where {T, Dist <: Distribution} = DistributionTrace{T, Dist}(val, args, logpdf(dist, val, args...), dist)
@inline DistributionTrace(val::T, args::Tuple, dist::Dist) where {T, Dist <: Distribution} = DistributionTrace{T, Dist}(val, args, logpdf(dist, val, args...), dist)

# we need to know the specific distribution in the trace type so the compiler can specialize GFI calls fully
@inline get_trace_type(::Dist) where {T, Dist <: Distribution{T}} = DistributionTrace{T, Dist}

function Base.convert(::Type{<:DistributionTrace{U, <:Any}}, tr::DistributionTrace{<:Any, Dist}) where {U, Dist}
    DistributionTrace{U, Dist}(convert(U, tr.val), tr.args, tr.score, tr.dist)
end

"""
    val::T = random(dist::Distribution{T}, args...)

Sample a random choice from the given distribution with the given arguments.
"""
function random end

"""
    lpdf = logpdf(dist::Distribution{T}, value::T, args...)

Evaluate the log probability (density) of the value.
"""
function logpdf end

"""
    has::Bool = has_output_grad(dist::Distribution)

Return true if the distribution computes the gradient of the logpdf with respect to the value of the random choice.
"""
function has_output_grad end

"""
    grads::Tuple = logpdf_grad(dist::Distribution{T}, value::T, args...)

Compute the gradient of the logpdf with respect to the value, and each of the arguments.

If `has_output_grad` returns false, then the first element of the returned tuple is `nothing`.
Otherwise, the first element of the tuple is the gradient with respect to the value.
If the return value of `has_argument_grads` has a false value for at position `i`, then the `i+1`th element of the returned tuple has value `nothing`.
Otherwise, this element contains the gradient with respect to the `i`th argument.
"""
function logpdf_grad end

is_discrete(::Distribution) = false # default

# NOTE: has_argument_grad is documented and exported in gen_fn_interface.jl

get_return_type(::Distribution{T}) where {T} = T


##############################
# Distribution GFI Interface #
##############################

@inline Base.getindex(trace::DistributionTrace) = trace.val
@inline Gen.get_args(trace::DistributionTrace) = trace.args
@inline Gen.get_choices(trace::DistributionTrace) = ValueChoiceMap(trace.val) # should be able to get type of val
@inline Gen.get_retval(trace::DistributionTrace) = trace.val
@inline Gen.get_gen_fn(trace::DistributionTrace) = dist(trace)
@inline Gen.get_score(trace::DistributionTrace) = trace.score
@inline Gen.project(trace::DistributionTrace, ::EmptySelection) = 0.
@inline Gen.project(trace::DistributionTrace, ::AllSelection) = get_score(trace)
@inline Gen.project(trace::DistributionTrace, c::ComplementSelection) = project_complement(trace, c.complement)
@inline project_complement(trace::DistributionTrace, ::EmptySelection) = get_score(trace)
@inline project_complement(trace::DistributionTrace, ::AllSelection) = 0.
@inline project_complement(trace::DistributionTrace, c::ComplementSelection) = Gen.project(trace, c.complement)
@inline function Gen.simulate(dist::Distribution, args::Tuple)
    val = random(dist, args...)
    DistributionTrace(val, args, dist)
end
@inline Gen.generate(dist::Distribution, args::Tuple, ::EmptyChoiceMap) = (simulate(dist, args), 0.)
@inline function Gen.generate(dist::Distribution, args::Tuple, constraints::ValueChoiceMap)
    tr = DistributionTrace(get_value(constraints), args, dist)
    weight = get_score(tr)
    (tr, weight)
end
@inline function Gen.update(tr::DistributionTrace, args::Tuple, argdiffs::Tuple, constraints::ValueChoiceMap)
    new_tr = DistributionTrace(get_value(constraints), args, dist(tr))
    weight = get_score(new_tr) - get_score(tr)
    (new_tr, weight, UnknownChange(), get_choices(tr))
end
@inline function Gen.update(tr::DistributionTrace, args::Tuple, argdiffs::Tuple, constraints::EmptyChoiceMap)
    new_tr = DistributionTrace(tr.val, args, dist(tr))
    weight = get_score(new_tr) - get_score(tr)
    (new_tr, weight, NoChange(), EmptyChoiceMap())
end
@inline Gen.update(tr::DistributionTrace, args::Tuple, argdiffs::NTuple{n, NoChange}, constraints::EmptyChoiceMap) where {n} = (tr, 0., NoChange())
# TODO: do I need an update method to handle empty choicemaps which are not `EmptyChoiceMap`s?
@inline Gen.regenerate(tr::DistributionTrace, args::Tuple, argdiffs::NTuple{n, NoChange}, selection::EmptySelection) where {n} = (tr, 0., NoChange())
@inline function Gen.regenerate(tr::DistributionTrace, args::Tuple, argdiffs::Tuple, selection::EmptySelection)
    new_tr = DistributionTrace(tr.val, args, dist(tr))
    weight = get_score(new_tr) - get_score(tr)
    (new_tr, weight, NoChange())
end
@inline function Gen.regenerate(tr::DistributionTrace, args::Tuple, argdiffs::Tuple, selection::AllSelection)
    new_tr = simulate(dist(tr), args)
    (new_tr, 0., UnknownChange())
end
@inline function Gen.propose(dist::Distribution, args::Tuple)
    val = random(dist, args...)
    score = logpdf(dist, val, args...)
    (ValueChoiceMap(val), score, val)
end
@inline function Gen.assess(dist::Distribution, args::Tuple, choices::ValueChoiceMap)
    weight = logpdf(dist, get_value(choices), args...)
    (weight, choices.val)
end


# Gradient-based methods
@inline Gen.accepts_output_grad(dist::Distribution) = has_output_grad(dist)

function Gen.choice_gradients(tr::DistributionTrace, ::AllSelection, retgrad)
    if !has_output_grad(dist(tr))
        error("Distribution $(dist(tr)) does not compute gradient of logpdf with respect to value")
    end
    grads = logpdf_grad(dist(tr), tr.val, tr.args...)
    output_grad = grads[1]
    arg_grads = grads[2:end]
    choice_values = ValueChoiceMap(tr.val)
    choice_grads = ValueChoiceMap(isnothing(retgrad) ? output_grad : output_grad .+ retgrad)
    return arg_grads, choice_values, choice_grads
end

@inline function Gen.choice_gradients(tr::DistributionTrace, ::EmptySelection, retgrad)
    arg_grads = logpdf_grad(dist(tr), tr.val, tr.args...)[2:end]
    choice_values = EmptyChoiceMap()
    choice_grads = EmptyChoiceMap()
    return arg_grads, choice_values, choice_grads
end

function Gen.choice_gradients(tr::DistributionTrace, c::ComplementSelection, retgrad)
    if c.complement isa EmptySelection
        return choice_gradients(tr, AllSelection(), retgrad)
    elseif c.complement isa AllSelection
        return choice_gradients(tr, EmptySelection(), retgrad)
    elseif c.complement isa ComplementSelection
        return choice_gradients(tr, c.complement.complement, retgrad)
    else
        error("Choice gradients not implemented for generic complement selection")
    end
end

@inline function Gen.accumulate_param_gradients!(tr::DistributionTrace, retgrad, scale_factor)
    arg_grads = logpdf_grad(dist(tr), tr.val, tr.args...)[2:end]
    return arg_grads
end


###########
# Exports #
###########

export Distribution
export random
export logpdf
export logpdf_grad
export has_output_grad
export is_discrete
