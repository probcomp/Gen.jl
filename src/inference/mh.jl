"""
    (new_trace, accepted) = default_mh(trace, selection::AddressSet)

Perform a Metropolis-Hastings update that proposes new values for the selected addresses from the internal proposal (often using ancestral sampling).
"""
function default_mh(trace, selection::AddressSet)
    args = get_args(trace)
    (new_trace, weight) = free_update(args, noargdiff, trace, selection)
    if log(rand()) < weight
        # accept
        return (new_trace, true)
    else
        # reject
        return (trace, false)
    end
end

"""
    (new_trace, accepted) = simple_mh(trace, proposal::GenerativeFunction, proposal_args::Tuple)

Perform a Metropolis-Hastings update that proposes new values for some subset of random choices in the given trace using the given proposal generative function.

The proposal generative function should take as its first argument the current trace of the model, and remaining arguments `proposal_args`.
All addresses sampled by the proposal must be in the existing model trace.
The proposal may modify the control flow of the model, but values of new addresses in the model are sampled from the model's internal proposal distribution.
"""
function simple_mh(trace, proposal::GenerativeFunction, proposal_args::Tuple)
    model_args = get_args(trace)
    proposal_args_forward = (trace, proposal_args...,)
    (fwd_assmt, fwd_weight, _) = propose(proposal, proposal_args_forward)
    (new_trace, weight, discard) = fix_update(
        model_args, noargdiff, trace, fwd_assmt)
    proposal_args_backward = (new_trace, proposal_args...,)
    (bwd_weight, _) = assess(proposal, proposal_args_backward, discard)
    alpha = weight - fwd_weight + bwd_weight
    if log(rand()) < alpha
        # accept
        return (new_trace, true)
    else
        # reject
        return (trace, false)
    end
end

"""
    (new_trace, accepted) = custom_mh(trace, proposal::GenerativeFunction, proposal_args::Tuple)

Perform a Metropolis-Hastings update that proposes new values for some subset of random choices in the given trace using the given proposal generative function.

The proposal generative function should take as its first argument the current trace of the model, and remaining arguments `proposal_args`.
If the proposal modifies addresses that determine the control flow in the model, values must be provided by the proposal for any addresses that are newly sampled by the model.
"""
function custom_mh(trace, proposal::GenerativeFunction, proposal_args::Tuple,
                   correction=(prev_trace, new_trace) -> 0.)
    model_args = get_args(trace)
    proposal_args_forward = (trace, proposal_args...,)
    (fwd_assmt, fwd_weight, _) = propose(proposal, proposal_args_forward)
    (new_trace, weight, discard) = force_update(
        model_args, noargdiff, trace, fwd_assmt)
    proposal_args_backward = (new_trace, proposal_args...,)
    (bwd_weight, _) = assess(proposal, proposal_args_backward, discard)
    alpha = weight - fwd_weight + bwd_weight
    alpha += correction(trace, new_trace)
    if log(rand()) < alpha
        # accept
        return (new_trace, true)
    else
        # reject
        return (trace, false)
    end
end

"""

    (new_trace, accepted) = general_mh(trace, proposal, proposal_args, bijection)

Apply a MCMC update constructed from a bijection and a proposal that satisfies detailed balance.
"""
function general_mh(trace, proposal::GenerativeFunction, proposal_args::Tuple,
                    bijection::Function)
    model = get_gen_fn(trace)
    model_args = get_args(trace)
    model_score = get_score(trace)
    (fwd_assmt, fwd_score, _) = propose(proposal, (trace, proposal_args...,))
    input = pair(get_assmt(trace), fwd_assmt, :model, :proposal)
    context = (model_args, proposal_args)
    (output, logabsdet) = bijection(input, context)
    (constraints, bwd_assmt) = unpair(output, :model, :proposal)
    (new_trace, _, _, _) = force_update(model_args, noargdiff, trace, constraints)
    new_model_score = get_score(new_trace)
    (bwd_score, _) = assess(proposal, (new_trace, proposal_args...), bwd_assmt)
    alpha = new_model_score - model_score - fwd_score + bwd_score + logabsdet
    if log(rand()) < alpha
        # accept
        (new_trace, true)
    else
        # reject
        (trace, false)
    end
end

export default_mh
export simple_mh
export custom_mh
export general_mh
