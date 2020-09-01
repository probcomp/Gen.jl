function sample_momenta(n::Int)
    Float64[random(normal, 0, 1) for _=1:n]
end

function assess_momenta(momenta)
    logprob = 0.
    for val in momenta
        logprob += logpdf(normal, val, 0, 1)
    end
    logprob
end

"""
    (new_trace, accepted) = hmc(
        trace, selection::Selection; L=10, eps=0.1,
        check=false, observations=EmptyChoiceMap())

Apply a Hamiltonian Monte Carlo (HMC) update that proposes new values for the selected addresses, returning the new trace (which is equal to the previous trace if the move was not accepted) and a `Bool` indicating whether the move was accepted or not.

Hamilton's equations are numerically integrated using leapfrog integration with step size `eps` for `L` steps. See equations (5.18)-(5.20) of Neal (2011).

# References
Neal, Radford M. (2011), "MCMC Using Hamiltonian Dynamics", Handbook of Markov Chain Monte Carlo, pp. 113-162. URL: http://www.mcmchandbook.net/HandbookChapter5.pdf
"""
function hmc(
        trace::U, selection::Selection; L=10, eps=0.1,
        check=false, observations=EmptyChoiceMap()) where {T,U}
    prev_model_score = get_score(trace)
    retval_grad = accepts_output_grad(get_gen_fn(trace)) ? zero(get_retval(trace)) : nothing

    # run leapfrog dynamics
    new_trace = trace
    (_, values_trie, gradient_trie) = choice_gradients(new_trace, selection, retval_grad)
    values = to_array(values_trie, Float64)
    gradient = to_array(gradient_trie, Float64)
    momenta = sample_momenta(length(values))
    prev_momenta_score = assess_momenta(momenta)
    for step=1:L

        # half step on momenta
        momenta += (eps / 2) * gradient

        # full step on positions
        values += eps * momenta

        # get new gradient
        values_trie = from_array(values_trie, values)
        (new_trace, _, _) = update(new_trace, values_trie)
        (_, _, gradient_trie) = choice_gradients(new_trace, selection, retval_grad)
        gradient = to_array(gradient_trie, Float64)

        # half step on momenta
        momenta += (eps / 2) * gradient
    end
    check && check_observations(get_choices(new_trace), observations)

    # assess new model score (negative potential energy)
    new_model_score = get_score(new_trace)

    # assess new momenta score (negative kinetic energy)
    new_momenta_score = assess_momenta(-momenta)

    # accept or reject
    alpha = new_model_score - prev_model_score + new_momenta_score - prev_momenta_score
    if log(rand()) < alpha
        (new_trace, true)
    else
        (trace, false)
    end
end

check_is_kernel(::typeof(hmc)) = true
is_custom_primitive_kernel(::typeof(hmc)) = false
reversal(::typeof(hmc)) = hmc

export hmc
