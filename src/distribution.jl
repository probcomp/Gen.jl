###############################
# Core Distribution Interface #
###############################

struct DistributionTrace{T, Dist} <: Trace
    val::T
    args
    score::Float64
end
@inline dist(::DistributionTrace{T, Dist}) where {T, Dist} = Dist()

abstract type Distribution{T} <: GenerativeFunction{T, DistributionTrace{T}} end
DistributionTrace{T, Dist}(val::T, args::Tuple, dist::Dist) where {T, Dist <: Distribution} = DistributionTrace{T, Dist}(val, args, logpdf(dist, val, args...))
@inline DistributionTrace(val::T, args::Tuple, dist::Dist) where {T, Dist <: Distribution} = DistributionTrace{T, Dist}(val, args, logpdf(dist, val, args...))

# we need to know the specific distribution in the trace type so the compiler can specialize GFI calls fully
@inline get_trace_type(::Dist) where {T, Dist <: Distribution{T}} = DistributionTrace{T, Dist}

function Base.convert(::Type{<:DistributionTrace{U, <:Any}}, tr::DistributionTrace{<:Any, Dist}) where {U, Dist}
    DistributionTrace{U, Dist}(convert(U, tr.val), tr.args, tr.score)
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
@inline Gen.get_choices(trace::DistributionTrace) = Value(trace.val) # should be able to get type of val
@inline Gen.get_retval(trace::DistributionTrace) = trace.val
@inline Gen.get_gen_fn(trace::DistributionTrace) = dist(trace)
@inline Gen.get_score(trace::DistributionTrace) = trace.score
@inline Gen.project(trace::DistributionTrace, ::EmptySelection) = 0.
@inline Gen.project(trace::DistributionTrace, ::AllSelection) = get_score(trace)

@inline function Gen.simulate(dist::Distribution, args::Tuple)
    val = random(dist, args...)
    DistributionTrace(val, args, dist)
end
@inline Gen.generate(dist::Distribution, args::Tuple, ::EmptyChoiceMap) = (simulate(dist, args), 0.)
@inline function Gen.generate(dist::Distribution, args::Tuple, constraints::Value)
    tr = DistributionTrace(get_value(constraints), args, dist)
    weight = get_score(tr)
    (tr, weight)
end
@inline function Gen.update(tr::DistributionTrace, args::Tuple, argdiffs::Tuple, spec::Value, ::AllSelection)
    new_tr = DistributionTrace(get_value(spec), args, dist(tr))
    weight = get_score(new_tr) - get_score(tr)
    (new_tr, weight, UnknownChange(), get_choices(tr))
end
@inline function Gen.update(tr::DistributionTrace, args::Tuple, argdiffs::Tuple, ::EmptyAddressTree, ::Selection)
    new_tr = DistributionTrace(tr.val, args, dist(tr))
    weight = get_score(new_tr) - get_score(tr)
    (new_tr, weight, NoChange(), EmptyChoiceMap())
end
@inline function Gen.update(tr::DistributionTrace, args::Tuple, argdiffs::Tuple{Vararg{NoChange}}, ::EmptyAddressTree, ::Selection)
    (tr, 0., NoChange(), EmptyChoiceMap())
end
@inline function Gen.update(tr::DistributionTrace, args::Tuple, argdiffs::Tuple, ::AllSelection, ::EmptyAddressTree)
    new_val = random(dist(tr), args...)
    new_tr = DistributionTrace(new_val, args, dist(tr))
    (new_tr, 0., UnknownChange(), get_choices(tr))
end
@inline function Gen.update(tr::DistributionTrace, args::Tuple, argdiffs::Tuple, ::AllSelection, ::AllSelection)
    new_val = random(dist(tr), args...)
    new_tr = DistributionTrace(new_val, args, dist(tr))
    (new_tr, -get_score(tr), UnknownChange(), get_choices(tr))
end
@inline function Gen.propose(dist::Distribution, args::Tuple)
    val = random(dist, args...)
    score = logpdf(dist, val, args...)
    (Value(val), score, val)
end
@inline function Gen.assess(dist::Distribution, args::Tuple, choices::Value)
    weight = logpdf(dist, get_value(choices), args...)
    (weight, choices.val)
end
@inline function Gen.assess(dist::Distribution, args::Tuple, ::EmptyChoiceMap)
    error("Call to `assess` did not provide a value constraint for a call to $dist.")
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
