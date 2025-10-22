# Momenta sampling with different metrics

function sample_momenta(n::Int)
    Float64[random(normal, 0, 1) for _=1:n]
end

function sample_momenta(n::Int, metric::AbstractVector)
    @assert all(>(0), metric) "All diagonal metric values must be positive"
    return sqrt.(metric) .* sample_momenta(n)
end

function sample_momenta(n::Int, metric::LinearAlgebra.Diagonal)
    sample_momenta(n::Int, LinearAlgebra.diag(metric))
end

function sample_momenta(n::Int, metric::AbstractMatrix)
    mvnormal(zeros(n), metric)
end

function sample_momenta(n::Int, metric::Nothing)
    sample_momenta(n)
end

# Assessing momenta log probabilities with different metrics

function assess_momenta(momenta)
    logprob = 0.
    for val in momenta
        logprob += logpdf(normal, val, 0, 1)
    end
    logprob
end

function assess_momenta(momenta, metric::AbstractVector)
    logprob = 0.
    for (val, m) in zip(momenta, metric)
        logprob += logpdf(normal, val, 0, sqrt(m))
    end
    logprob
end

function assess_momenta(momenta, metric::LinearAlgebra.Diagonal)
    assess_momenta(momenta, LinearAlgebra.diag(metric))
end

function assess_momenta(momenta, metric::AbstractMatrix)
    logpdf(mvnormal, momenta, zeros(length(momenta)), metric)
end

function assess_momenta(momenta, metric::Nothing)
    assess_momenta(momenta)
end

"""
    (new_trace, accepted) = hmc(
        trace, selection::Selection; L=10, eps=0.1,
        check=false, observations=EmptyChoiceMap(), metric = nothing)

Apply a Hamiltonian Monte Carlo (HMC) update that proposes new values for the 
selected addresses, returning the new trace (which is equal to the previous trace
if the move was not accepted) and a `Bool` indicating whether the move was accepted or not.

Hamilton's equations are numerically integrated using leapfrog integration with
step size `eps` for `L` steps and initial momenta sampled from a Gaussian distribution with
covariance given by `metric` (mass matrix). 

Sampling with HMC is improved by using a metric/mass matrix that approximates the 
**inverse** covariance of the target distribution, and is equivalent to a linear transformation
of the parameter space (see Neal, 2011). The following options are supported for `metric`:

- `nothing` (default): identity matrix
- `Vector`: diagonal matrix with the given vector as the diagonal
- `Diagonal`: diagonal matrix lowers to the vector of the diagonal entries
- `Matrix`: dense matrix

See equations (5.18)-(5.20) of Neal (2011).

# References
Neal, Radford M. (2011), "MCMC Using Hamiltonian Dynamics", Handbook of Markov Chain Monte Carlo, pp. 113-162. URL: http://www.mcmchandbook.net/HandbookChapter5.pdf
"""
function hmc(
        trace::Trace, selection::Selection; L=10, eps=0.1,
        check=false, observations=EmptyChoiceMap(), metric = nothing)
    prev_model_score = get_score(trace)
    args = get_args(trace)
    retval_grad = accepts_output_grad(get_gen_fn(trace)) ? zero(get_retval(trace)) : nothing
    argdiffs = map((_) -> NoChange(), args)

    # run leapfrog dynamics
    new_trace = trace
    (_, values_trie, gradient_trie) = choice_gradients(new_trace, selection, retval_grad)
    values = to_array(values_trie, Float64)
    gradient = to_array(gradient_trie, Float64)
    momenta = sample_momenta(length(values), metric)
    prev_momenta_score = assess_momenta(momenta, metric)
    for step=1:L

        # half step on momenta
        momenta += (eps / 2) * gradient

        # full step on positions
        values += eps * momenta

        # get new gradient
        values_trie = from_array(values_trie, values)
        (new_trace, _, _) = update(new_trace, args, argdiffs, values_trie)
        (_, _, gradient_trie) = choice_gradients(new_trace, selection, retval_grad)
        gradient = to_array(gradient_trie, Float64)

        # half step on momenta
        momenta += (eps / 2) * gradient
    end
    check && check_observations(get_choices(new_trace), observations)

    # assess new model score (negative potential energy)
    new_model_score = get_score(new_trace)

    # assess new momenta score (negative kinetic energy)
    new_momenta_score = assess_momenta(-momenta, metric)

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
