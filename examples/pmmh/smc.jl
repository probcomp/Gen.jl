import Distributions

function logsumexp(arr)
    min_arr = maximum(arr)
    min_arr + log(sum(exp.(arr .- min_arr)))
end


"""
A scheme for sequential Monte Carlo inference in a state space model, without rejuvenation.
Concerete subtypes should implement the following methods:
    init(scheme::StateSpaceSMCScheme{H})
    init_score(scheme::StateSpaceSMCScheme{H}, state::H)
    forward(scheme::StateSpaceSMCScheme{H}, prev_state::H, t::Integer)
    forward_score(scheme::StateSpaceSMCScheme{H}, prev_state::H, state::H, t::Integer)
    get_num_steps(scheme::StateSpaceSMCScheme{H})
    get_num_particles(scheme::StateSpaceSMCScheme{H})
    get_ess_threshold(scheme::StateSpaceSMCScheme{H})
"""
abstract type StateSpaceSMCScheme{H} end

function init end
function init_score end
function forward end
function forward_score end
function get_num_steps end
function get_num_particles end
function get_ess_threshold end

struct StateSpaceSMCResult{H}
    states::Matrix{H}
    parents::Matrix{Int}
    log_weights::Vector{Float64}
    log_ml_estimate::Float64
    num_resamples::Int
end

function effective_sample_size(log_weights::Vector{Float64})
    # assumes weights are normalized
    log_ess = -logsumexp(2. * log_weights)
    exp(log_ess)
end

"""
Sequential Monte Carlo for state space models without rejuvenation and with multinomial resampling.
See algoritihm 3.1.1 of:
Del Moral, Pierre, Arnaud Doucet, and Ajay Jasra. "Sequential monte carlo samplers."
Journal of the Royal Statistical Society: Series B (Statistical Methodology) 68.3 (2006): 411-436.
Also see algorithm 2.2.1 of:
Andrieu, Christophe, Arnaud Doucet, and Roman Holenstein. "Particle markov chain monte carlo methods."
Journal of the Royal Statistical Society: Series B (Statistical Methodology) 72.3 (2010): 269-342.
"""
function smc(scheme::StateSpaceSMCScheme{H}) where {H}
    N = get_num_particles(scheme)
    T = get_num_steps(scheme)
    ess_threshold = get_ess_threshold(scheme)
    states = Matrix{H}(undef, N, T)
    parents = Matrix{Int}(undef, N, T-1)
    log_unnormalized_weights = Vector{Float64}(undef, N)
    log_ml_estimate = 0.
    for i=1:N
        (states[i, 1], log_unnormalized_weights[i]) = init(scheme)
    end

    num_resamples = 0
    for t=2:T
        log_total_weight = logsumexp(log_unnormalized_weights)
        log_normalized_weights = log_unnormalized_weights .- log_total_weight
        if effective_sample_size(log_normalized_weights) < ess_threshold
            weights = exp.(log_normalized_weights)
            parents[:, t-1] = rand(Distributions.Categorical(weights / sum(weights)), N)
            log_ml_estimate += log_total_weight - log(N)
            log_unnormalized_weights = zeros(N)
            num_resamples += 1
        else
            parents[:, t-1] = 1:N
        end
        for i=1:N
            parent = parents[i, t-1]
            (states[i, t], log_incremental_weight) = forward(scheme, states[parent, t-1], t)
            log_unnormalized_weights[i] += log_incremental_weight
        end
    end
    log_total_weight = logsumexp(log_unnormalized_weights)
    log_normalized_weights = log_unnormalized_weights .- log_total_weight
    log_ml_estimate += log_total_weight - log(N)
    StateSpaceSMCResult(states, parents, log_normalized_weights, log_ml_estimate, num_resamples)
end

# TODO modify conditional SMC below

const ONE = 1
"""
Conditional sequential Monte Carlo update for state space models without rejuvenation and with multinomial resampling.
See sections 2.4.3 and 4.3 of:
Andrieu, Christophe, Arnaud Doucet, and Roman Holenstein. "Particle markov chain monte carlo methods."
Journal of the Royal Statistical Society: Series B (Statistical Methodology) 72.3 (2010): 269-342.
"""

function conditional_smc(scheme::StateSpaceSMCScheme{H}, distinguished_particle::Vector{H}) where {H}
    N = get_num_particles(scheme)
    T = get_num_steps(scheme)
    ess_threshold = get_ess_threshold(scheme)
    if length(distinguished_particle) != T
        error("Expected particle length $T, actual length was $(length(distinguished_particle))")
    end
    states = Matrix{H}(undef, N, T)
    parents = Matrix{Int}(undef, N, T-1)
    log_unnormalized_weights = Vector{Float64}(undef, N)
    log_ml_estimate = 0.

    # Due to symmetries, the ancestral indices of the distinguished
    # particle do not matter, so we set them all to 1.
    # Note that this may not suffice for other resampling schemes.
    states[ONE, 1] = distinguished_particle[1]
    log_unnormalized_weights[ONE] = init_score(scheme, states[ONE, 1])

    for i=2:N
        (states[i, 1], log_unnormalized_weights[i]) = init(scheme)
    end

    num_resamples = 0
    for t=2:T
        log_total_weight = logsumexp(log_unnormalized_weights)
        log_normalized_weights = log_unnormalized_weights .- log_total_weight
        if effective_sample_size(log_normalized_weights) < ess_threshold
            weights = exp.(log_normalized_weights)
            parents[:, t-1] = rand(Distributions.Categorical(weights / sum(weights)), N)
            log_ml_estimate += log_total_weight - log(N)
            log_unnormalized_weights = zeros(N)
            num_resamples += 1
        else
            parents[:, t-1] = 1:N
        end

        # handle distinguished particle
        parents[ONE, t-1] = ONE
        states[ONE, t] = distinguished_particle[t]
        log_unnormalized_weights[ONE] += forward_score(scheme, states[ONE, t-1], states[ONE, t], t)

        for i=2:N
            parent = parents[i, t-1]
            (states[i, t], log_incremental_weight) = forward(scheme, states[parent, t-1], t)
            log_unnormalized_weights[i] += log_incremental_weight
        end
    end
    log_total_weight = logsumexp(log_unnormalized_weights)
    log_normalized_weights = log_unnormalized_weights .- log_total_weight
    log_ml_estimate += log_total_weight - log(N)
    StateSpaceSMCResult(states, parents, log_normalized_weights, log_ml_estimate, num_resamples)
end

function get_particle(result::StateSpaceSMCResult{H}, final_index::Integer) where {H}
    (N, T) = size(result.states)
    particle = Vector{H}(undef, T)
    particle[T] = result.states[final_index, T]
    current_index = final_index
    for t=T-1:-1:1
        current_index = result.parents[current_index, t]
        particle[t] = result.states[current_index, t]
    end
    return particle
end
