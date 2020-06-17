###############################
# Core Distribution Interface #
###############################

struct DistributionTrace{T, Dist} <: Trace
    val::T
    args
    dist::Dist
end

abstract type Distribution{T} <: GenerativeFunction{T, DistributionTrace{T}} end

function Base.convert(::Type{DistributionTrace{U, <:Any}}, tr::DistributionTrace{<:Any, Dist}) where {U, Dist}
    DistributionTrace{U, Dist}(convert(U, tr.val), tr.args, tr.dist)
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

Return true of the gradient if the distribution computes the gradient of the logpdf with respect to the value of the random choice.
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

function is_discrete end

# NOTE: has_argument_grad is documented and exported in gen_fn_interface.jl

get_return_type(::Distribution{T}) where {T} = T


##############################
# Distribution GFI Interface #
##############################

@inline Base.getindex(trace::DistributionTrace) = trace.val
@inline Gen.get_args(trace::DistributionTrace) = trace.args
@inline Gen.get_choices(trace::DistributionTrace) = ValueChoiceMap(trace.val) # should be able to get type of val
@inline Gen.get_retval(trace::DistributionTrace) = trace.val
@inline Gen.get_gen_fn(trace::DistributionTrace) = trace.dist

# TODO: for performance would it be better to store the score in the trace?
@inline Gen.get_score(trace::DistributionTrace) = logpdf(trace.dist, trace.val, trace.args...)
@inline Gen.project(trace::DistributionTrace, ::EmptySelection) = 0.
@inline Gen.project(trace::DistributionTrace, ::AllSelection) = get_score(trace)

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
    new_tr = DistributionTrace(get_value(constraints), args, tr.dist)
    weight = get_score(new_tr) - get_score(tr)
    (new_tr, weight, UnknownChange(), get_choices(tr))
end
@inline function Gen.update(tr::DistributionTrace, args::Tuple, argdiffs::Tuple, constraints::EmptyChoiceMap)
    new_tr = DistributionTrace(tr.val, args, tr.dist)
    weight = get_score(new_tr) - get_score(tr)
    (new_tr, weight, NoChange(), EmptyChoiceMap())
end
# TODO: do I need an update method to handle empty choicemaps which are not `EmptyChoiceMap`s?
@inline Gen.regenerate(tr::DistributionTrace, args::Tuple, argdiffs::NTuple{n, NoChange}, selection::EmptySelection) where {n} = (tr, 0., NoChange())
@inline function Gen.regenerate(tr::DistributionTrace, args::Tuple, argdiffs::Tuple, selection::EmptySelection)
    new_tr = DistributionTrace(tr.val, args, tr.dist)
    weight = get_score(new_tr) - get_score(tr)
    (new_tr, weight, NoChange())
end
@inline function Gen.regenerate(tr::DistributionTrace, args::Tuple, argdiffs::Tuple, selection::AllSelection)
    new_val = random(tr.dist, args...)
    new_tr = DistributionTrace(new_val, args, tr.dist)
    (new_tr, 0., UnknownChange())
end
@inline function Gen.propose(dist::Distribution, args::Tuple)
    val = random(dist, args...)
    score = logpdf(dist, val, args...)
    (ValueChoiceMap(val), score, val)
end
@inline function Gen.assess(dist::Distribution, args::Tuple, choices::ValueChoiceMap)
    weight = logpdf(dist, choices.val, args...)
    (weight, choices.val)
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
