using PyPlot
using Gen

import Random
Random.seed!(1)
include("poisson_process.jl")

# Example from Section 4 of Reversible jump Markov chain Monte Carlo
# computation and Bayesian model determination 

#########
# model #
#########

const K = :k
const EVENTS = :events
const UNSORTED_CHANGEPT = :unsorted_changept
const UNSORTED_RATE = :unsorted_rate

function sort_change_pts(unsorted_change_pts::Vector)
    k = length(unsorted_change_pts)
    sorted_to_unsorted = sortperm(unsorted_change_pts)
    unsorted_to_sorted = Vector{Int}(undef, k)
    for i in 1:k
        unsorted_to_sorted[sorted_to_unsorted[i]] = i
    end
    sorted_change_pts = unsorted_change_pts[sorted_to_unsorted]
    return (sorted_change_pts, unsorted_to_sorted, sorted_to_unsorted)
end

function test_sort()
    unsorted_change_pts = [0.3, 0.1, 0.2, 0.0, -1.0]
    (change_pts, unsorted_to_sorted, sorted_to_unsorted) = sort_change_pts(unsorted_change_pts)
    @assert change_pts == [-1.0, 0.0, 0.1, 0.2, 0.3]
    @assert sorted_to_unsorted == [5, 4, 2, 3, 1]
    @assert unsorted_to_sorted == [5, 3, 4, 2, 1]
end

test_sort()

@gen function model(T::Float64)

    # prior on number of change points
    k = @trace(poisson(3.), K)

    # prior distribution on change points
    (sorted_change_pts, unsorted_to_sorted, sorted_to_unsorted) = sort_change_pts(
        Float64[({(UNSORTED_CHANGEPT, i)} ~ uniform(0, T)) for i in 1:k])

    # k + 1 rate values
    # unsorted_rates[i] for i > 0 is the rate of the segment immediately before
    # the i'th unsorted changepoint. unsorted_rates[k+1] is the rate of the
    # last segment
    alpha = 1.0; beta = 200.0
    unsorted_rates = Float64[@trace(Gen.gamma(alpha, 1. / beta), (UNSORTED_RATE, i)) for i=1:k+1]
    sorted_rates = Vector{Float64}(undef, k+1)
    sorted_rates[1:k] = unsorted_rates[sorted_to_unsorted]
    sorted_rates[k+1] = unsorted_rates[k+1]

    # poisson process
    bounds = vcat([0.], sorted_change_pts, [T])
    @trace(piecewise_poisson_process(bounds, sorted_rates), EVENTS)

    return (sorted_change_pts, unsorted_to_sorted, sorted_to_unsorted, sorted_rates)
end

get_t_min(trace) = 0.0
get_t_max(trace) = get_args(trace)[1]
get_num_change_pts(trace) = trace[:k]
get_sorted_change_pts(trace) = get_retval(trace)[1]
get_sorted_rates(trace) = get_retval(trace)[4]

function render(trace; ymax=0.02)
    k = get_num_change_pts(trace)
    bounds = vcat([get_t_min(trace)], get_sorted_change_pts(trace), [get_t_max(trace)])
    rates = get_sorted_rates(trace)
    for i=1:length(rates)
        lower = bounds[i]
        upper = bounds[i+1]
        rate = rates[i]
        plot([lower, upper], [rate, rate], color="black", linewidth=2)
    end
    points = trace[EVENTS]
    scatter(points, -rand(length(points)) * (ymax/5.), color="black", s=5)
    ax = gca()
    xlim = [get_t_min(trace), get_t_max(trace)]
    plot(xlim, [0., 0.], "--")
    ax[:set_xlim](xlim)
    ax[:set_ylim](-ymax/5., ymax)
end

function show_prior_samples()
    figure(figsize=(16,16))
    T = 40000.
    for i=1:16
        println("simulating $i")
        subplot(4, 4, i)
        (trace, ) = generate(model, (T,), EmptyChoiceMap())
        render(trace; ymax=0.015)
    end
    tight_layout(pad=0)
    savefig("prior_samples.pdf")
end


#############
# rate move #
#############

@gen function rate_proposal(trace)

    # pick a random segment whose rate to change
    i = @trace(uniform_discrete(1, trace[K]+1), :i)

    # propose new value for the rate
    #cur_rate = trace[(UNSORTED_RATE, i)]
    #I@trace(uniform_continuous(cur_rate/2., cur_rate*2.), :new_rate)
    new_rate_scaled ~ uniform(0, 1)

    nothing
end

# it is an involution because it:
# - maintains i = fwd_choices[:i] constant
# - swaps choices[(RATE, i)] with fwd_choices[:new_rate]

@bijection function rate_involution(model_args, proposal_args, proposal_retval)
    i = @read_discrete_from_proposal(:i)
    @write_discrete_to_proposal(:i, i)
    #new_rate = @read_continuous_from_proposal(:new_rate)
    new_rate_scaled = @read_continuous_from_proposal(:new_rate_scaled)
    cur_rate = @read_continuous_from_model((UNSORTED_RATE, i))
    lower_bound = cur_rate / 2.0
    upper_bound = cur_rate * 2.0
    new_rate = lower_bound + new_rate_scaled * (upper_bound - lower_bound)
    @write_continuous_to_model((UNSORTED_RATE, i), new_rate)
    
    #prev_rate = @read_continuous_from_model((UNSORTED_RATE, i))
    prev_rate_scaled = (cur_rate - (new_rate / 2.0)) / (new_rate * 2.0 - new_rate / 2.0)
    @write_continuous_to_proposal(:new_rate_scaled, prev_rate_scaled)
end

is_involution!(rate_involution)

rate_move(trace) = metropolis_hastings(trace, rate_proposal, (), rate_involution, check=false)

#################
# position move #
#################

function get_sorted_idx(trace, unsorted_idx::Int)
    (sorted_change_pts, unsorted_to_sorted, sorted_to_unsorted, _) = get_retval(trace)
    return unsorted_to_sorted[unsorted_idx]
end

function get_lower_bound(trace, unsorted_idx::Int)
    if get_sorted_idx(trace, unsorted_idx) == 1
        return get_t_min(trace)
    else
        return sorted_change_pts[sorted_idx - 1]
    end
end

function get_upper_bound(trace, unsorted_idx::Int)
    if get_sorted_idx(trace, unsorted_idx) == get_num_change_pts(trace)
        return get_t_max(trace)
    else
        return sorted_change_pts[sorted_idx + 1]
    end
end

# TODO, note: the density is not nonzero everywhere with respect to the reference measure...
# but it could easily be made so by sampling from uniform(0, 1) the location between lower and upper bound...

@gen function position_proposal(trace)
    k = trace[K]
    @assert k > 0

    # pick a random unsorted changepoint to change
    i = @trace(uniform_discrete(1, k), :i) 

    sorted_idx = get_sorted_idx(trace, i)
    sorted_to_unsorted = get_retval(trace)[3]
    #lower = get_lower_bound(trace, i)
    #upper = get_upper_bound(trace, i)
    @trace(uniform_continuous(0, 1), :new_changept_scaled)
    #@trace(uniform_continuous(lower, upper), :new_changept)

    return (sorted_idx, sorted_to_unsorted)
end

# it is an involution because it:
# - maintains i = fwd_choices[:i] constant
# - swaps choices[(CHANGEPT, i)] with fwd_choices[:new_changept]

@bijection function position_involution(model_args, proposal_args, proposal_retval)
    (sorted_idx, sorted_to_unsorted) = proposal_retval
    k = @read_discrete_from_model(:k)
    i = @read_discrete_from_proposal(:i)
    @write_discrete_to_proposal(:i, i)
    new_changept_scaled = @read_continuous_from_proposal(:new_changept_scaled)
    lower_bound = sorted_idx == 1 ? 0.0 : @read_continuous_from_model((UNSORTED_CHANGEPT, sorted_to_unsorted[sorted_idx-1]))
    upper_bound = sorted_idx == k ? T : @read_continuous_from_model((UNSORTED_CHANGEPT, sorted_to_unsorted[sorted_idx+1]))
    new_changept = lower_bound + new_changept_scaled * (upper_bound - lower_bound)
    @write_continuous_to_model((UNSORTED_CHANGEPT, i), new_changept)


    prev_changept = @read_continuous_from_model((UNSORTED_CHANGEPT, i))
    prev_changept_scaled = (prev_changept - lower_bound) / (upper_bound - lower_bound)
    @write_continuous_to_proposal(:new_changept_scaled, prev_changept_scaled)

    #@copy_model_to_proposal((UNSORTED_CHANGEPT, i), :new_changept)
    #@copy_proposal_to_model(:new_changept, (UNSORTED_CHANGEPT, i))
end

is_involution!(position_involution)

position_move(trace) = metropolis_hastings(trace, position_proposal, (), position_involution, check=false)


######################
# birth / death move #
######################

const CHOSEN = :chosen
const IS_BIRTH = :is_birth
const NEW_CHANGEPT = :new_changept
const U = :u

@gen function birth_death_proposal(trace)
    #T = get_t_max(trace)
    k = trace[K]
    # if k = 0, then always do a birth move
    # if k > 0, then randomly choose a birth or death move
    isbirth = (k == 0) ? true : @trace(bernoulli(0.5), IS_BIRTH)
    if isbirth
        {NEW_CHANGEPT} ~ uniform_continuous(get_t_min(trace), get_t_max(trace))
        {U} ~ uniform_continuous(0.0, 1.0)
    end
    (_, unsorted_to_sorted, sorted_to_unsorted, _) = get_retval(trace)
    return (unsorted_to_sorted, sorted_to_unsorted) # NOTE: based on testing predicates on real values
end

function new_rates(cur_rate, u, cur_cp, prev_cp, next_cp)
    d_prev = cur_cp - prev_cp
    d_next = next_cp - cur_cp
    @assert d_prev > 0
    @assert d_next > 0
    d_total = d_prev + d_next
    log_cur_rate = log(cur_rate)
    log_ratio = log(1 - u) - log(u)
    prev_rate = exp(log_cur_rate - (d_next / d_total) * log_ratio)
    next_rate = exp(log_cur_rate + (d_prev / d_total) * log_ratio)
    @assert prev_rate > 0.
    @assert next_rate > 0.
    (prev_rate, next_rate)
end

function new_rates_inverse(prev_rate, next_rate, cur_cp, prev_cp, next_cp)
    d_prev = cur_cp - prev_cp
    d_next = next_cp - cur_cp
    @assert d_prev > 0
    @assert d_next > 0
    d_total = d_prev + d_next
    log_prev_rate = log(prev_rate)
    log_next_rate = log(next_rate)
    cur_rate = exp((d_prev / d_total) * log_prev_rate + (d_next / d_total) * log_next_rate)
    u = prev_rate / (prev_rate + next_rate)
    @assert cur_rate > 0.
    (cur_rate, u)
end

# it is an involution because:
# - it switches back and forth between birth move and death move
# - it maintains fwd_choices[CHOSEN] constant (applying it twice will first insert a new
#   changepoint and then remove that same changepoint)
# - new_rates, curried on cp_new, cp_prev, and cp_next, is the inverse of new_rates_inverse.

@bijection function birth_death_involution(model_args, proposal_args, proposal_retval)
    T = model_args[1]
    (unsorted_to_sorted, sorted_to_unsorted) = proposal_retval

    # current number of changepoints
    k = @read_discrete_from_model(K)
    
    # if k == 0, then we can only do a birth move
    isbirth = (k == 0) || @read_discrete_from_proposal(IS_BIRTH)

    # if we are a birth move, the inverse is a death move
    if k > 1 || isbirth
        @write_discrete_to_proposal(IS_BIRTH, !isbirth)
    end
    
    if isbirth
        @bijcall(birth(k, unsorted_to_sorted, sorted_to_unsorted))
    else
        @bijcall(death(k, unsorted_to_sorted, sorted_to_unsorted))
    end
end

@bijection function birth(k::Int, unsorted_to_sorted, sorted_to_unsorted)
    k = @read_discrete_from_model(K)
    @write_discrete_to_model(K, k+1)
    cp_new = @read_continuous_from_proposal(NEW_CHANGEPT)

    # set new changepoint
    @copy_proposal_to_model(NEW_CHANGEPT, (UNSORTED_CHANGEPT, k+1))

    # re-sort the changepoints, with the new one added
    unsorted_change_pts = [@read_continuous_from_model((UNSORTED_CHANGEPT, i)) for i in 1:k]
    push!(unsorted_change_pts, cp_new)
    (sorted_change_pts, unsorted_to_sorted, sorted_to_unsorted) = sort_change_pts(unsorted_change_pts)
    sorted_idx = unsorted_to_sorted[k+1]

    # the unsorted index of the next rate choice
    prev_next_unsorted_rate_idx = (sorted_idx == k+1) ? k+1 : sorted_to_unsorted[sorted_idx+1]
    new_next_unsorted_rate_idx = (sorted_idx == k+1) ? k+2 : sorted_to_unsorted[sorted_idx+1]

    # the previous and next changepoints
    cp_prev = (sorted_idx == 1) ? 0.0 : sorted_change_pts[sorted_idx-1]
    cp_next = (sorted_idx == k+1) ? T : sorted_change_pts[sorted_idx+1]

    # compute new rates
    h_cur = @read_continuous_from_model((UNSORTED_RATE, prev_next_unsorted_rate_idx))
    u = @read_continuous_from_proposal(U)
    (h_prev, h_next) = new_rates(h_cur, u, cp_new, cp_prev, cp_next)

    # set new rates
    @write_continuous_to_model((UNSORTED_RATE, k+1), h_prev)
    @write_continuous_to_model((UNSORTED_RATE, new_next_unsorted_rate_idx), h_next)

    # shift the final rate up, only if it is not where we inserted it...
    if sorted_idx != k+1
        @copy_model_to_model((UNSORTED_RATE, k+1), (UNSORTED_RATE, k+2))
    end
end

# TODO rename the last rate to 'last_rate' so we don't need to shift it around...

@bijection function death(k::Int, unsorted_to_sorted, sorted_to_unsorted)
    k = @read_discrete_from_model(K)
    @write_discrete_to_model(K, k-1)
    cp_deleted = @read_continuous_from_model((UNSORTED_CHANGEPT, k)) # NOTE: will not be copied...

    # set proposed changepoint
    @copy_model_to_proposal((UNSORTED_CHANGEPT, k), NEW_CHANGEPT) # but gets copied here..

    # re-sort the changepoints (we could use the result of the sorting from the model trace, if we wanted)
    unsorted_change_pts = [@read_continuous_from_model((UNSORTED_CHANGEPT, i)) for i in 1:k] # all get copied..
    (sorted_change_pts, unsorted_to_sorted, sorted_to_unsorted) = sort_change_pts(unsorted_change_pts)
    sorted_idx = unsorted_to_sorted[k]

    # the unsorted index of the next rate choice
    prev_next_unsorted_rate_idx = (sorted_idx == k) ? k+1 : sorted_to_unsorted[sorted_idx+1]
    new_next_unsorted_rate_idx = (sorted_idx == k) ? k : sorted_to_unsorted[sorted_idx+1]
    @assert prev_next_unsorted_rate_idx != k

    # the previous and next changepoints
    cp_prev = (sorted_idx == 1) ? 0.0 : sorted_change_pts[sorted_idx-1]
    cp_next = (sorted_idx == k) ? T : sorted_change_pts[sorted_idx+1]

    # compute cur rate and u
    h_prev = @read_continuous_from_model((UNSORTED_RATE, k))
    h_next = @read_continuous_from_model((UNSORTED_RATE, prev_next_unsorted_rate_idx))
    (h_cur, u) = new_rates_inverse(h_prev, h_next, cp_deleted, cp_prev, cp_next)
    @write_continuous_to_proposal(U, u)

    # set cur rate
    @write_continuous_to_model((UNSORTED_RATE, new_next_unsorted_rate_idx), h_cur)

    # shift the final rate, only if we are not deleting the last one..
    if sorted_idx != k
        @copy_model_to_model((UNSORTED_RATE, k+1), (UNSORTED_RATE, k))
    end
end

is_involution!(birth_death_involution)

#struct UniformPerm <: Gen.Distribution{Vector{Int}} end
#const uniform_permutation = UniformPerm()
#import Random, SpecialFunctions
#Gen.random(::UniformPerm, n::Int) = Random.randperm(n)
#Gen.logpdf(::UniformPerm, perm::Vector{Int}, n::Int) = isperm(perm) ? -SpecialFunctions.logfactorial(n) : -Inf
#Gen.logpdf_grad(::UniformPerm, perm::Vector{Int}, n::Int) = (nothing, nothing)
#Gen.is_discrete(::UniformPerm) = true
#Gen.has_output_grad(::UniformPerm) = false
#Gen.has_argument_grads(::UniformPerm) = (false,)

@gen function permutation_proposal(trace)
    k = get_num_change_pts(trace)
    to_swap ~ uniform_discrete(1, k-1)
end

@bijection function permutation_involution(model_args, proposal_args, proposal_retval)
    k = @read_discrete_from_model(:k)
    to_swap_with_k = @read_discrete_from_proposal(:to_swap)
    @copy_proposal_to_proposal(:to_swap, :to_swap)
    @copy_model_to_model((UNSORTED_RATE, k), (UNSORTED_RATE, to_swap_with_k))
    @copy_model_to_model((UNSORTED_RATE, to_swap_with_k), (UNSORTED_RATE, k))
    @copy_model_to_model((UNSORTED_CHANGEPT, k), (UNSORTED_CHANGEPT, to_swap_with_k))
    @copy_model_to_model((UNSORTED_CHANGEPT, to_swap_with_k), (UNSORTED_CHANGEPT, k))
end

is_involution!(permutation_involution)

function birth_death_move(trace)
    if trace[:k] > 1
        trace, acc = metropolis_hastings(trace, permutation_proposal, (), permutation_involution, check=false)
        @assert acc
    end
    return metropolis_hastings(trace, birth_death_proposal, (), birth_death_involution, check=false)
end

function mcmc_step(trace)
    (trace, _) = rate_move(trace)
    if trace[K] > 0
        (trace, _) = position_move(trace)
    end
    (trace, _) = birth_death_move(trace)
    trace
end

function simple_mcmc_step(trace)
    (trace, _) = rate_move(trace)
    if trace[K] > 0
        (trace, _) = position_move(trace)
    end
    (trace, _) = metropolis_hastings(trace, k_selection)
    trace
end

function do_mcmc(T, num_steps::Int)
    (trace, _) = generate(model, (T,), observations)
    for iter=1:num_steps
        k = trace[K]
        if iter % 1000 == 0
            println("iter $iter of $num_steps, k: $k")
        end
        trace = mcmc_step(trace)
    end
    trace
end

const k_selection = select(K)

function do_simple_mcmc(T, num_steps::Int)
    (trace, _) = generate(model, (T,), observations)
    for iter=1:num_steps
        k = trace[K]
        if iter % 1000 == 0
            println("iter $iter of $num_steps, k: $k")
        end
        trace = simple_mcmc_step(trace)
    end
    trace
end


########################
# inference experiment #
########################

import Random
Random.seed!(1)

# load data set
import CSV
function load_data_set()
    df = CSV.read("$(@__DIR__)/coal.csv")
    dates = df[1]
    dates = dates .- minimum(dates)
    dates * 365.25 # convert years to days
end

const points = load_data_set()
const T = maximum(points)
const observations = choicemap()
observations[EVENTS] = points

function show_posterior_samples()
    figure(figsize=(16,16))
    for i=1:16
        println("replicate $i")
        subplot(4, 4, i)
        trace = do_mcmc(T, 200)
        render(trace; ymax=0.015)
    end
    tight_layout(pad=0)
    savefig("posterior_samples.pdf")

    figure(figsize=(16,16))
    for i=1:16
        println("replicate $i")
        subplot(4, 4, i)
        trace = do_simple_mcmc(T, 200)
        render(trace; ymax=0.015)
    end
    tight_layout(pad=0)
    savefig("posterior_samples_simple.pdf")
end

function get_rate_vector(trace, test_points)
    k = trace[K]
    cps = get_sorted_change_pts(trace)
    hs = get_sorted_rates(trace)
    rate = Vector{Float64}()
    cur_h_idx = 1
    cur_h = hs[cur_h_idx]
    next_cp_idx = 1
    upper = (next_cp_idx == k + 1) ? T : cps[next_cp_idx]
    for x in test_points
        while x > upper
            next_cp_idx += 1
            upper = (next_cp_idx == k + 1) ? T : cps[next_cp_idx]
            cur_h_idx += 1
            cur_h = hs[cur_h_idx]
        end
        push!(rate, cur_h)
    end
    rate
end

# compute posterior mean rate curve

function plot_posterior_mean_rate()
    test_points = collect(1.0:10.0:T)
    rates = Vector{Vector{Float64}}()
    num_samples = 0
    num_steps = 8000
    for reps=1:20
        (trace, _) = generate(model, (T,), observations)
        for iter=1:num_steps
            if iter % 1000 == 0
                println("iter $iter of $num_steps, k: $(trace[K])")
            end
            trace = mcmc_step(trace)
            if iter > 4000
                num_samples += 1
                rate_vector = get_rate_vector(trace, test_points)
                @assert length(rate_vector) == length(test_points)
                push!(rates, rate_vector)
            end
        end
    end
    posterior_mean_rate = zeros(length(test_points))
    for rate in rates
        posterior_mean_rate += rate / Float64(num_samples)
    end
    ymax = 0.010
    figure()
    plot(test_points, posterior_mean_rate, color="black")
    scatter(points, -rand(length(points)) * (ymax/6.), color="black", s=5)
    ax = gca()
    xlim = [0., T]
    plot(xlim, [0., 0.], "--")
    ax[:set_xlim](xlim)
    ax[:set_ylim](-ymax/5., ymax)
    savefig("posterior_mean_rate.pdf")
end

function plot_trace_plot()
    figure(figsize=(8, 4))

    # reversible jump
    (trace, _) = generate(model, (T,), observations)
    rate1 = Float64[]
    num_clusters_vec = Int[]
    burn_in = 0
    for iter=1:burn_in + 1000 
        trace = mcmc_step(trace)
        if iter > burn_in
            push!(num_clusters_vec, trace[K])
        end
    end
    subplot(2, 1, 1)
    plot(num_clusters_vec, "b")

    # simple MCMC
    (trace, _) = generate(model, (T,), observations)
    rate1 = Float64[]
    num_clusters_vec = Int[]
    burn_in = 0
    for iter=1:burn_in + 1000 
        trace = simple_mcmc_step(trace)
        if iter > burn_in
            push!(num_clusters_vec, trace[K])
        end
    end
    subplot(2, 1, 2)
    plot(num_clusters_vec, "b")

    savefig("trace_plot.pdf")
end

println("showing prior samples...")
show_prior_samples()

println("showing posterior samples...")
show_posterior_samples()

println("estimating posterior mean rate...")
plot_posterior_mean_rate()

println("making trace plot...")
plot_trace_plot()
