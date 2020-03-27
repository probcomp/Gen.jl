function metropolis_hastings end

check_is_kernel(::typeof(metropolis_hastings)) = true
is_custom_primitive_kernel(::typeof(metropolis_hastings)) = false
reversal(::typeof(metropolis_hastings)) = metropolis_hastings

"""
    (new_trace, accepted) = metropolis_hastings(
        trace, selection::Selection;
        check=false, observations=EmptyChoiceMap())

Perform a Metropolis-Hastings update that proposes new values for the selected addresses from the internal proposal (often using ancestral sampling), returning the new trace (which is equal to the previous trace if the move was not accepted) and a Bool indicating whether the move was accepted or not.
"""
function metropolis_hastings(
        trace, selection::Selection;
        check=false, observations=EmptyChoiceMap())
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    (new_trace, weight) = regenerate(trace, args, argdiffs, selection)
    check && check_observations(get_choices(new_trace), observations)
    if log(rand()) < weight
        # accept
        return (new_trace, true)
    else
        # reject
        return (trace, false)
    end
end


"""
    (new_trace, accepted) = metropolis_hastings(
        trace, proposal::GenerativeFunction, proposal_args::Tuple;
        check=false, observations=EmptyChoiceMap())

Perform a Metropolis-Hastings update that proposes new values for some subset of random choices in the given trace using the given proposal generative function, returning the new trace (which is equal to the previous trace if the move was not accepted) and a Bool indicating whether the move was accepted or not.

The proposal generative function should take as its first argument the current trace of the model, and remaining arguments `proposal_args`.
If the proposal modifies addresses that determine the control flow in the model, values must be provided by the proposal for any addresses that are newly sampled by the model.
"""
function metropolis_hastings(
        trace, proposal::GenerativeFunction, proposal_args::Tuple;
        check=false, observations=EmptyChoiceMap())
    model_args = get_args(trace)
    argdiffs = map((_) -> NoChange(), model_args)
    proposal_args_forward = (trace, proposal_args...,)
    (fwd_choices, fwd_weight, _) = propose(proposal, proposal_args_forward)
    (new_trace, weight, _, discard) = update(trace,
        model_args, argdiffs, fwd_choices)
    proposal_args_backward = (new_trace, proposal_args...,)
    (bwd_weight, _) = assess(proposal, proposal_args_backward, discard)
    alpha = weight - fwd_weight + bwd_weight
    check && check_observations(get_choices(new_trace), observations)
    if log(rand()) < alpha
        # accept
        return (new_trace, true)
    else
        # reject
        return (trace, false)
    end
end

"""
    (new_trace, accepted) = metropolis_hastings(
        trace, proposal::GenerativeFunction, proposal_args::Tuple, involution;
        check=false, observations=EmptyChoiceMap())

Perform a generalized (reversible jump) Metropolis-Hastings update based on an involution (bijection that is its own inverse) on a space of choice maps, returning the new trace (which is equal to the previous trace if the move was not accepted) and a Bool indicating whether the move was accepted or not.

The `involution` is a callable value that has the following signature:

    (new_trace, bwd_choices::ChoiceMap, weight) = involution(trace, fwd_choices::ChoiceMap, fwd_ret, proposal_args::Tuple)

Most users will want to construct `involution` using the [Involution DSL](@ref) with the [`@involution`](@ref) macro, but is also possible to provide a Julia function for `involution`.

The generative function `proposal` is executed on arguments `(trace, proposal_args...)`, producing a choice map `fwd_choices` and return value `fwd_ret`.
For each value of model arguments (contained in `trace`) and `proposal_args`, the `involution` function applies an involution that maps the tuple `(get_choices(trace), fwd_choices)` to the tuple `(get_choices(new_trace), bwd_choices)`.
Note that `fwd_ret` is a deterministic function of `fwd_choices` and `proposal_args`.
When only discrete random choices are used, the `weight` must be equal to `get_score(new_trace) - get_score(trace)`.
When continuous random choices are used, the `weight` returned by the involution must include an additive term that is the determinant of the the Jacobian of the bijection on the continuous random choices that is obtained by currying the involution on the discrete random choices.
"""
function metropolis_hastings(
        trace, proposal::GenerativeFunction,
        proposal_args::Tuple, involution;
        check=false, observations=EmptyChoiceMap())
    (fwd_choices, fwd_score, fwd_ret) = propose(proposal, (trace, proposal_args...,))
    (new_trace, bwd_choices, weight) = involution(trace, fwd_choices, fwd_ret, proposal_args)
    (bwd_score, bwd_ret) = assess(proposal, (new_trace, proposal_args...), bwd_choices)
    check && check_observations(get_choices(new_trace), observations)
    if check
        (trace_rt, fwd_choices_rt, weight_rt) = involution(new_trace, bwd_choices, bwd_ret, proposal_args)
        if !isapprox(fwd_choices_rt, fwd_choices)
            println("fwd_choices:")
            println(fwd_choices)
            println("fwd_choices_rt:")
            println(fwd_choices_rt)
            error("Involution round trip check failed")
        end
        if !isapprox(get_choices(trace), get_choices(trace_rt))
            println("get_choices(trace):")
            println(get_choices(trace))
            println("get_choices(trace_rt):")
            println(get_choices(trace_rt))
            error("Involution round trip check failed")
        end
        if !isapprox(weight, -weight_rt)
            println("weight: $weight, -weight_rt: $(-weight_rt)")
            error("Involution round trip check failed")
        end
    end
    if log(rand()) < weight - fwd_score + bwd_score
        # accept
        (new_trace, true)
    else
        # reject
        (trace, false)
    end
end

"""
    (new_trace, accepted) = mh(trace, selection::Selection; ..)
    (new_trace, accepted) = mh(trace, proposal::GenerativeFunction, proposal_args::Tuple; ..)
    (new_trace, accepted) = mh(trace, proposal::GenerativeFunction, proposal_args::Tuple, involution::Function; ..)

Alias for [`metropolis_hastings`](@ref). Perform a Metropolis-Hastings update on the given trace.
"""
const mh = metropolis_hastings

export metropolis_hastings, mh
