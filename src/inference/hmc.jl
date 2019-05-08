function sample_momenta(mass, n::Int)
    Float64[random(normal, 0, mass) for _=1:n]
end

function assess_momenta(momenta, mass)
    logprob = 0.
    for val in momenta
        logprob += logpdf(normal, val, 0, mass)
    end
    logprob
end

"""
    (new_trace, accepted) = hmc(trace, selection::Selection, mass=0.1, L=10, eps=0.1)

Apply a Hamiltonian Monte Carlo (HMC) update.

Neal, Radford M. "MCMC using Hamiltonian dynamics." Handbook of Markov Chain Monte Carlo 2.11 (2011): 2.

[Reference URL](http://www.mcmchandbook.net/HandbookChapter5.pdf)
"""
function hmc(trace::U, selection::Selection;
             mass=0.1, L=10, eps=0.1) where {T,U}
    prev_model_score = get_score(trace)
    args = get_args(trace)
    retval_grad = accepts_output_grad(get_gen_fn(trace)) ? zero(get_retval(trace)) : nothing
    argdiffs = map((_) -> NoChange(), args)

    # run leapfrog dynamics
    new_trace = trace
    local prev_momenta_score::Float64
    local momenta::Vector{Float64}
    for step=1:L

        # half step on momenta
        (_, values_trie, gradient_trie) = choice_gradients(new_trace, selection, retval_grad)
        values = to_array(values_trie, Float64)
        gradient = to_array(gradient_trie, Float64)
        if step == 1
            momenta = sample_momenta(mass, length(values))
            prev_momenta_score = assess_momenta(momenta, mass)
        else
            momenta += (eps / 2) * gradient
        end

        # full step on positions
        values_trie = from_array(values_trie, values + eps * momenta)

        # half step on momenta
        (new_trace, _, _) = update(new_trace, args, argdiffs, values_trie)
        (_, _, gradient_trie) = choice_gradients(new_trace, selection, retval_grad)
        gradient = to_array(gradient_trie, Float64)
        momenta += (eps / 2) * gradient
    end

    # assess new model score (negative potential energy)
    new_model_score = get_score(new_trace)

    # assess new momenta score (negative kinetic energy)
    new_momenta_score = assess_momenta(-momenta, mass)

    # accept or reject
    alpha = new_model_score - prev_model_score + new_momenta_score - prev_momenta_score
    if log(rand()) < alpha
        (new_trace, true)
    else
        (trace, false)
    end
end

export hmc
