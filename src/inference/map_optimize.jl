"""
    new_trace = map_optimize(trace, selection::Selection,
        max_step_size=0.1, tau=0.5, min_step_size=1e-16, verbose=false)

Perform backtracking gradient ascent to optimize the log probability of the trace over selected continuous choices.

Selected random choices must have support on the entire real line.
"""
function map_optimize(trace, selection::Selection;
                      max_step_size=0.1, tau=0.5, min_step_size=1e-16, verbose=false)
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    retval_grad = accepts_output_grad(get_gen_fn(trace)) ? zero(get_retval(trace)) : nothing

    (_, values, gradient) = choice_gradients(trace, selection, retval_grad)
    values_vec = to_array(values, Float64)
    gradient_vec = to_array(gradient, Float64)
    step_size = max_step_size
    score = get_score(trace)
    while true
        new_values_vec = values_vec + gradient_vec * step_size
        values = from_array(values, new_values_vec)
        # TODO discard and weight are not actually needed, there should be a more specialized variant
        (new_trace, _, _, discard) = update(trace, args, argdiffs, values)
        new_score = get_score(new_trace)
        change = new_score - score
        if verbose
            println("step_size: $step_size, prev score: $score, new score: $new_score, change: $change")
        end
        if change >= 0.
            # it got better, return it
            return new_trace
        elseif step_size < min_step_size
            # it got worse, but we ran out of attempts
            return trace
        end

        # try again with a smaller step size
        step_size = tau * step_size
    end
end

export map_optimize
