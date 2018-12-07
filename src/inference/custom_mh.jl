function custom_mh(model::GenerativeFunction{T,U},
                   proposal::GenerativeFunction,
                   proposal_args::Tuple, trace::U,
                   correction=(prev_trace, new_trace) -> 0.) where {T,U}
    model_args = get_call_record(trace).args
    proposal_args_forward = (trace, proposal_args...,)
    (fwd_assmt, fwd_weight, _) = propose(proposal, proposal_args_forward)
    (new_trace, weight, discard) = force_update(
        model, model_args, noargdiff, trace, fwd_assmt)
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

export custom_mh
