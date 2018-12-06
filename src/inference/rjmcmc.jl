function rjmcmc(model, forward, forward_args_rest, backward, backward_args_rest,
                injective, injective_args, trace, correction)
    model_args = get_call_record(trace).args
    model_score = get_call_record(trace).score
    forward_args = (trace, forward_args_rest...,)
    forward_trace = simulate(forward, forward_args)
    forward_score = get_call_record(forward_trace).score
    input = pair(get_assignment(trace), get_assignment(forward_trace), :model, :proposal)
    (output, logabsdet) = apply(injective, injective_args, input)
    (model_constraints, backward_constraints) = unpair(output, :model, :proposal)
    new_trace = assess(model, model_args, model_constraints)
    new_model_score = get_call_record(new_trace).score
    backward_args = (new_trace, backward_args_rest...,)
    backward_trace = assess(backward, backward_args, backward_constraints)
    backward_score = get_call_record(backward_trace).score
    alpha = new_model_score - model_score - forward_score + backward_score + logabsdet + correction(new_trace)
    if log(rand()) < alpha
        # accept
        return new_trace
    else
        # reject
        return trace
    end
end

export rjmcmc
