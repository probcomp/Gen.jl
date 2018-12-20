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
export custom_mh
export general_mh
