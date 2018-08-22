########
# MCMC #
########

function mh(model::Generator, proposal::Generator, proposal_args::Tuple, trace)
    model_args = get_call_record(trace).args
    forward_trace = simulate(proposal, proposal_args, Some(get_choices(trace)))
    forward_score = get_call_record(forward_trace).score
    constraints = get_choices(forward_trace)
    (new_trace, weight, discard) = update(
        model, model_args, NoChange(), trace, constraints)
    backward_trace = assess(proposal, proposal_args, discard, Some(get_choices(new_trace)))
    backward_score = get_call_record(backward_trace).score
    if log(rand()) < weight - forward_score + backward_score
        # accept
        return new_trace
    else
        # reject
        return trace
    end
end

function mh(model::Generator, selector::SelectionFunction, selector_args::Tuple, trace)
    (selection, _) = select(selector, selector_args, get_choices(trace))
    model_args = get_call_record(trace).args
    (new_trace, weight) = regenerate(model, model_args, NoChange(), trace, selection)
    if log(rand()) < weight
        # accept
        return new_trace
    else
        # reject
        return trace
    end
end

export mh
