function custom_mh(model::GenerativeFunction, proposal::GenerativeFunction, proposal_args_rest::Tuple,
                   trace, correction=(prev_trace, new_trace) -> 0.; verbose=false)
    model_args = get_call_record(trace).args
    proposal_args_forward = (trace, proposal_args_rest...,)
    forward_trace = simulate(proposal, proposal_args_forward)
    forward_score = get_call_record(forward_trace).score
    constraints = get_assignment(forward_trace)
    (new_trace, weight, discard) = update(
        model, model_args, noargdiff, trace, constraints)
    proposal_args_backward = (new_trace, proposal_args_rest...,)
    backward_trace = assess(proposal, proposal_args_backward, discard)
    backward_score = get_call_record(backward_trace).score
    alpha = weight - forward_score + backward_score
    alpha += correction(trace, new_trace)
    if log(rand()) < alpha
        verbose && println("accept")
        # accept
        return new_trace
    else
        # reject
        verbose && println("reject")
        return trace
    end
end

export custom_mh
