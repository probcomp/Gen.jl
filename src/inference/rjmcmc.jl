function rjmcmc(trace, proposal, proposal_args, bijection)
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

export rjmcmc
