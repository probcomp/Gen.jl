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
    # TODO add a round trip check
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
        trace, proposal::GenerativeFunction, proposal_args::Tuple,
        involution::Union{TraceTransformDSLProgram,Function};
        check=false, observations=EmptyChoiceMap())

Perform a generalized (reversible jump) Metropolis-Hastings update based on an involution (bijection that is its own inverse) on a space of choice maps, returning the new trace (which is equal to the previous trace if the move was not accepted) and a Bool indicating whether the move was accepted or not.

Most users will want to construct `involution` using the [Trace Transform DSL](@ref) with the [`@transform`](@ref) macro, but for more user control it is also possible to provide a Julia function for `involution`, that has the following signature:

    (new_trace, bwd_choices::ChoiceMap, weight) = involution(trace::Trace, fwd_choices::ChoiceMap, fwd_retval, fwd_args::Tuple)

The generative function `proposal` is executed on arguments `(trace, proposal_args...)`, producing a choice map `fwd_choices` and return value `fwd_ret`.
For each value of model arguments (contained in `trace`) and `proposal_args`, the `involution` function applies an involution that maps the tuple `(get_choices(trace), fwd_choices)` to the tuple `(get_choices(new_trace), bwd_choices)`.
Note that `fwd_ret` is a deterministic function of `fwd_choices` and `proposal_args`.
When only discrete random choices are used, the `weight` must be equal to `get_score(new_trace) - get_score(trace)`.

When continuous random choices are used, the `weight` returned by the involution must include an additive correction term that is the determinant of the the Jacobian of the bijection on the continuous random choices that is obtained by currying the involution on the discrete random choices (this correction term is automatically computed if the involution is constructed using the [Trace Transform DSL](@ref)).
NOTE: The Jacobian matrix of the bijection on the continuous random choices must be full-rank (i.e. nonzero determinant).
The `check` keyword argument to the involution can be used to enable or disable any dynamic correctness checks that the involution performs; for successful executions, `check` does not alter the return value.
"""
function metropolis_hastings(
        trace, proposal::GenerativeFunction,
        proposal_args::Tuple, involution::Union{TraceTransformDSLProgram,Function};
        check=false, observations=EmptyChoiceMap())
    trace_translator = SymmetricTraceTranslator(proposal, proposal_args, involution)
    (new_trace, log_weight) = trace_translator(trace; check=check, observations=observations)
    if log(rand()) < log_weight
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
    (new_trace, accepted) = mh(trace, proposal::GenerativeFunction, proposal_args::Tuple, involution; ..)

Alias for [`metropolis_hastings`](@ref). Perform a Metropolis-Hastings update on the given trace.
"""
const mh = metropolis_hastings

"""
    (new_trace, accepted) = involutive_mcmc(trace, proposal::GenerativeFunction, proposal_args::Tuple, involution; ..)
    
Alias for the involutive form of [`metropolis_hastings`](@ref).
"""
function involutive_mcmc(trace, proposal, proposal_args, involution; check=false, observations=EmptyChoiceMap())
    return metropolis_hastings(
        trace, proposal, proposal_args, involution;
        check=check, observations=observations)
end

export metropolis_hastings, mh, involutive_mcmc
