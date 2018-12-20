
"""
    (new_trace, accepted) = mala(trace, selection::AddressSet, tau::Real)

Apply a Metropolis-Adjusted Langevin Algorithm (MALA) update.

[Reference URL](https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm)
"""
function mala(trace, selection::AddressSet, tau::Real)
    model_args = get_args(trace)
    std = sqrt(2 * tau)

    # forward proposal
    (_, values_trie, gradient_trie) = backprop_trace(trace, selection, nothing)
    values = to_array(values_trie, Float64)
    gradient = to_array(gradient_trie, Float64)
    forward_mu = values + tau * gradient
    forward_score = 0.
    proposed_values = Vector{Float64}(undef, length(values))
    for i=1:length(values)
        proposed_values[i] = random(normal, forward_mu[i], std)
        forward_score += logpdf(normal, proposed_values[i], forward_mu[i], std)
    end

    # evaluate model weight
    constraints = from_array(values_trie, proposed_values)
    (new_trace, weight, discard) = force_update(
        model_args, noargdiff, trace, constraints)

    # backward proposal
    (_, _, backward_gradient_trie) = backprop_trace(new_trace, selection, nothing)
    backward_gradient = to_array(backward_gradient_trie, Float64)
    @assert length(backward_gradient) == length(values)
    backward_score = 0.
    backward_mu  = proposed_values + tau * backward_gradient
    for i=1:length(values)
        backward_score += logpdf(normal, values[i], backward_mu[i], std)
    end

    # accept or reject
    alpha = weight - forward_score + backward_score
    if log(rand()) < alpha
        (new_trace, true)
    else
        (trace, false)
    end
end

export mala
