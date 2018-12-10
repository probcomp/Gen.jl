"""
    map_optimize

Backtracking gradient ascent for MAP inference on selected real-valued choices
"""
function map_optimize(selection::AddressSet,
                      trace; max_step_size=0.1, tau=0.5, min_step_size=1e-16, verbose=false)
    model_args = get_args(trace)
    (_, values, gradient) = backprop_trace(trace, selection, nothing)
    values_vec = to_array(values, Float64)
    gradient_vec = to_array(gradient, Float64)
    step_size = max_step_size
    score = get_score(trace)
    while true
        new_values_vec = values_vec + gradient_vec * step_size
        values = from_array(values, new_values_vec)
        # TODO discard and weight are not actually needed, there should be a more specialized variant
        (new_trace, _, discard, _) = force_update(model_args, noargdiff, trace, values)
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
