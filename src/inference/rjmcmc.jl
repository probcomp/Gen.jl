function rjmcmc(trace, forward, fwd_args, backward, bwd_args,
                injective, injective_args, correction)
    model = get_gen_fn(trace)
    model_args = get_args(trace)
    model_score = get_score(trace)
    (fwd_assmt, fwd_score) = propose(forward, (trace, fwd_args...,))
    input = pair(get_assmt(trace), fwd_assmt, :model, :proposal)
    (output, logabsdet) = apply(injective, injective_args, input)
    (model_constraints, bwd_assmt) = unpair(output, :model, :proposal)
    (new_trace, new_model_score) = initialize(model, model_args, model_constraints)
    (bwd_score, _) = assess(backward, (new_trace, bwd_args...), bwd_assmt)
    alpha = new_model_score - model_score - fwd_score + bwd_score + logabsdet + correction(new_trace)
    if log(rand()) < alpha
        # accept
        (new_trace, true)
    else
        # reject
        (trace, false)
    end
end

export rjmcmc
