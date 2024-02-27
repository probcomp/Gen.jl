
"""
    (new_trace, accepted) = mala(
        trace, selection::Selection, tau::Real;
        check=false, observations=EmptyChoiceMap(),
        rng::Random.AbstractRNG=Random.default_rng())

Apply a Metropolis-Adjusted Langevin Algorithm (MALA) update.

[Reference URL](https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm)
"""
function mala(
    trace, selection::Selection, tau::Real;
    check=false, observations=EmptyChoiceMap(),
    rng::Random.AbstractRNG=Random.default_rng()
)
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    std = sqrt(2 * tau)
    retval_grad = accepts_output_grad(get_gen_fn(trace)) ? zero(get_retval(trace)) : nothing

    # forward proposal
    (_, values_trie, gradient_trie) = choice_gradients(trace, selection, retval_grad)
    values = to_array(values_trie, Float64)
    gradient = to_array(gradient_trie, Float64)
    forward_mu = values + tau * gradient
    forward_score = 0.
    proposed_values = Vector{Float64}(undef, length(values))
    for i=1:length(values)
        proposed_values[i] = random(rng, normal, forward_mu[i], std)
        forward_score += logpdf(normal, proposed_values[i], forward_mu[i], std)
    end

    # evaluate model weight
    constraints = from_array(values_trie, proposed_values)
    (new_trace, weight, _, discard) = update(rng, trace,
        args, argdiffs, constraints)
    check && check_observations(get_choices(new_trace), observations)

    # backward proposal
    (_, _, backward_gradient_trie) = choice_gradients(new_trace, selection, retval_grad)
    backward_gradient = to_array(backward_gradient_trie, Float64)
    @assert length(backward_gradient) == length(values)
    backward_score = 0.
    backward_mu  = proposed_values + tau * backward_gradient
    for i=1:length(values)
        backward_score += logpdf(normal, values[i], backward_mu[i], std)
    end

    # accept or reject
    alpha = weight - forward_score + backward_score
    if log(rand(rng)) < alpha
        (new_trace, true)
    else
        (trace, false)
    end
end

check_is_kernel(::typeof(mala)) = true
is_custom_primitive_kernel(::typeof(mala)) = false
reversal(::typeof(mala)) = mala

export mala
