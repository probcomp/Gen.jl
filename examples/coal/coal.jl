using PyPlot
using Gen

include("poisson_process.jl")

# Example from Section 4 of Reversible jump Markov chain Monte Carlo
# computation and Bayesian model determination

# minimum of k draws from uniform_continuous(lower, upper)

# we can sequentially sample the order statistics of a collection of K uniform
# continuous samples on the interval [a, b], by:
# x1 ~ min_uniform_continuous(a, b, K)
# x2 | x1 ~ min_uniform_continuous(x1, b, K-1)
# ..
# xK | x1 .. x_{K-1} ~ min_uniform_continuous(x_{K-1}, b, 1)

struct MinUniformContinuous <: Distribution{Float64} end
const min_uniform_continuous = MinUniformContinuous()

function Gen.logpdf(::MinUniformContinuous, x::Float64, lower::T, upper::U, k::Int) where {T<:Real,U<:Real}
    if x > lower && x < upper
        (k-1) * log(upper - x) + log(k) - k * log(upper - lower)
    else
        -Inf
    end
end

function Gen.random(::MinUniformContinuous, lower::T, upper::U, k::Int) where {T<:Real,U<:Real}
    # inverse CDF method
    p = rand()
    upper - (upper - lower) * (1. - p)^(1. / k)
end

#########
# model #
#########

const K = :k
const EVENTS = :events
const CHANGEPT = :changept
const RATE = :rate

@gen function model(T::Float64)

    # prior on number of change points
    k = @trace(poisson(3.), K)

    # prior on the location of (sorted) change points
    change_pts = Vector{Float64}(undef, k)
    lower = 0.
    for i=1:k
        cp = @trace(min_uniform_continuous(lower, T, k-i+1), (CHANGEPT, i))
        change_pts[i] = cp
        lower = cp
    end

    # k + 1 rate values
    # h$i is the rate for cp$(i-1) to cp$i where cp0 := 0 and where cp$(k+1) := T
    alpha = 1.
    beta = 200.
    rates = Float64[@trace(Gen.gamma(alpha, 1. / beta), (RATE, i)) for i=1:k+1]

    # poisson process
    bounds = vcat([0.], change_pts, [T])
    @trace(piecewise_poisson_process(bounds, rates), EVENTS)
end

function render(trace; ymax=0.02)
    T = get_args(trace)[1]
    k = trace[:k]
    bounds = vcat([0.], sort([trace[(CHANGEPT, i)] for i=1:k]), [T])
    rates = [trace[(RATE, i)] for i=1:k+1]
    for i=1:length(rates)
        lower = bounds[i]
        upper = bounds[i+1]
        rate = rates[i]
        plot([lower, upper], [rate, rate], color="black", linewidth=2)
    end
    points = trace[EVENTS]
    scatter(points, -rand(length(points)) * (ymax/5.), color="black", s=5)
    ax = gca()
    xlim = [0., T]
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
    cur_rate = trace[(RATE, i)]
    @trace(uniform_continuous(cur_rate/2., cur_rate*2.), :new_rate)

    nothing
end

# it is an involution because it:
# - maintains i = fwd_choices[:i] constant
# - swaps choices[(RATE, i)] with fwd_choices[:new_rate]

@involution function rate_involution(model_args, proposal_args, proposal_retval)
    i = @read_discrete_from_proposal(:i)
    @write_discrete_to_proposal(:i, i)
    new_rate = @read_continuous_from_proposal(:new_rate)
    @write_continuous_to_model((RATE, i), new_rate)
    prev_rate = @read_continuous_from_model((RATE, i))
    @write_continuous_to_proposal(:new_rate, prev_rate)
end

rate_move(trace) = metropolis_hastings(trace, rate_proposal, (), rate_involution, check=true)

#################
# position move #
#################

@gen function position_proposal(trace)
    k = trace[K]
    @assert k > 0

    # pick a random changepoint to change
    i = @trace(uniform_discrete(1, k), :i)

    lower = (i == 1) ? 0. : trace[(CHANGEPT, i-1)]
    upper = (i == k) ? T : trace[(CHANGEPT, i+1)]
    @trace(uniform_continuous(lower, upper), :new_changept)

    i
end

# it is an involution because it:
# - maintains i = fwd_choices[:i] constant
# - swaps choices[(CHANGEPT, i)] with fwd_choices[:new_changept]

@involution function position_involution(model_args, proposal_args, proposal_retval::Int)
    i = @read_discrete_from_proposal(:i)
    @write_discrete_to_proposal(:i, i)
    @copy_model_to_proposal((CHANGEPT, i), :new_changept)
    @copy_proposal_to_model(:new_changept, (CHANGEPT, i))
end

position_move(trace) = metropolis_hastings(trace, position_proposal, (), position_involution, check=true)


######################
# birth / death move #
######################

const CHOSEN = :chosen
const IS_BIRTH = :is_birth
const NEW_CHANGEPT = :new_changept
const U = :u

@gen function birth_death_proposal(trace)
    T = get_args(trace)[1]
    k = trace[K]

    # if k = 0, then always do a birth move
    # if k > 0, then randomly choose a birth or death move
    isbirth = (k == 0) ? true : @trace(bernoulli(0.5), IS_BIRTH)

    if isbirth
        # pick the segment in which to insert the new changepoint
        # changepoints before move:  | 1     2    3 |
        # new changepoint (i = 2):   |    *         |
        # changepoints after move:   | 1  2  3    4 |
        i = @trace(uniform_discrete(1, k+1), CHOSEN)
        lower = (i == 1) ? 0. : trace[(CHANGEPT, i-1)]
        upper = (i == k+1) ? T : trace[(CHANGEPT, i)]
        @trace(uniform_continuous(lower, upper), NEW_CHANGEPT)
        @trace(uniform_continuous(0., 1.), U)
    else
        # pick the changepoint to be deleted
        # changepoints before move:     | 1  2  3    4 |
        # deleted changepoint (i = 2):  |    *         |
        # changepoints after move:      | 1     2    3 |
        @trace(uniform_discrete(1, k), CHOSEN)
    end
    nothing
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

@involution function birth_death_involution(model_args, proposal_args, proposal_retval::Nothing)
    T = model_args[1]

    # current number of changepoints
    k = @read_discrete_from_model(K)

    # if k == 0, then we can only do a birth move
    isbirth = (k == 0) || @read_discrete_from_proposal(IS_BIRTH)

    # if we are a birth move, the inverse is a death move
    if k > 1 || isbirth
        @write_discrete_to_proposal(IS_BIRTH, !isbirth)
    end

    # the changepoint to be added or deleted
    i = @read_discrete_from_proposal(CHOSEN)
    @copy_proposal_to_proposal(CHOSEN, CHOSEN)

    if isbirth
        @invcall(birth(k, i))
    else
        @invcall(death(k, i))
    end
end

@involution function birth(k::Int, i::Int)
    @write_discrete_to_model(K, k+1)

    cp_new = @read_continuous_from_proposal(NEW_CHANGEPT)
    cp_prev = (i == 1) ? 0. : @read_continuous_from_model((CHANGEPT, i-1))
    cp_next = (i == k+1) ? T : @read_continuous_from_model((CHANGEPT, i))

    # set new changepoint
    @copy_proposal_to_model(NEW_CHANGEPT, (CHANGEPT, i))

    # shift up changepoints
    for j=i+1:k+1
        @copy_model_to_model((CHANGEPT, j-1), (CHANGEPT, j))
    end

    # compute new rates
    h_cur = @read_continuous_from_model((RATE, i))
    u = @read_continuous_from_proposal(U)
    (h_prev, h_next) = new_rates(h_cur, u, cp_new, cp_prev, cp_next)

    # set new rates
    @write_continuous_to_model((RATE, i), h_prev)
    @write_continuous_to_model((RATE, i+1), h_next)

    # shift up rates
    for j=i+2:k+2
        @copy_model_to_model((RATE, j-1), (RATE, j))
    end
end

@involution function death(k::Int, i::Int)
    @write_discrete_to_model(K, k-1)

    cp_deleted = @read_continuous_from_model((CHANGEPT, i))
    cp_prev = (i == 1) ? 0. : @read_continuous_from_model((CHANGEPT, i-1))
    cp_next = (i == k) ? T : @read_continuous_from_model((CHANGEPT, i+1))
    @copy_model_to_proposal((CHANGEPT, i), NEW_CHANGEPT)

    # shift down changepoints
    for j=i:k-1
        @copy_model_to_model((CHANGEPT, j+1), (CHANGEPT, j))
    end

    # compute cur rate and u
    h_prev = @read_continuous_from_model((RATE, i))
    h_next = @read_continuous_from_model((RATE, i+1))
    (h_cur, u) = new_rates_inverse(h_prev, h_next, cp_deleted, cp_prev, cp_next)
    @write_continuous_to_proposal(U, u)

    # set cur rate
    @write_continuous_to_model((RATE, i), h_cur)

    # shift down rates
    for j=i+1:k
        @copy_model_to_model((RATE, j+1), (RATE, j))
    end
end

#function birth_death_involution(trace, fwd_choices::ChoiceMap, fwd_ret, proposal_args::Tuple)
    #model_args = get_args(trace)
    #T = model_args[1]
#
    #bwd_choices = choicemap()
#
    ## current number of changepoints
    #k = trace[K]
    #
    ## if k == 0, then we can only do a birth move
    #isbirth = (k == 0) || fwd_choices[IS_BIRTH]
#
    ## if we are a birth move, the inverse is a death move
    #if k > 1 || isbirth
        #bwd_choices[IS_BIRTH] = !isbirth
    #end
    #
    ## the changepoint to be added or deleted
    #i = fwd_choices[CHOSEN]
    #bwd_choices[CHOSEN] = i
#
    ## populate constraints
    #constraints = choicemap()
    #if isbirth
        #constraints[K] = k + 1
#
        #cp_new = fwd_choices[NEW_CHANGEPT]
        #cp_prev = (i == 1) ? 0. : trace[(CHANGEPT, i-1)]
        #cp_next = (i == k+1) ? T : trace[(CHANGEPT, i)]
#
        ## set new changepoint
        #constraints[(CHANGEPT, i)] = cp_new
#
        ## shift up changepoints
        #for j=i+1:k+1
            #constraints[(CHANGEPT, j)] = trace[(CHANGEPT, j-1)]
        #end
#
        ## compute new rates
        #h_cur = trace[(RATE, i)]
        #u = fwd_choices[U]
        #(h_prev, h_next) = new_rates([h_cur, u, cp_new, cp_prev, cp_next])
        #J = jacobian(new_rates, [h_cur, u, cp_new, cp_prev, cp_next])[:,1:2]
#
        ## set new rates
        #constraints[(RATE, i)] = h_prev
        #constraints[(RATE, i+1)] = h_next
#
        ## shift up rates
        #for j=i+2:k+2
            #constraints[(RATE, j)] = trace[(RATE, j-1)]
        #end
    #else
        #constraints[K] = k - 1
#
        #cp_deleted = trace[(CHANGEPT, i)]
        #cp_prev = (i == 1) ? 0. : trace[(CHANGEPT, i-1)]
        #cp_next = (i == k) ? T : trace[(CHANGEPT, i+1)]
        #bwd_choices[NEW_CHANGEPT] = cp_deleted
#
        ## shift down changepoints
        #for j=i:k-1
            #constraints[(CHANGEPT, j)] = trace[(CHANGEPT, j+1)]
        #end
#
        ## compute cur rate and u
        #h_prev = trace[(RATE, i)]
        #h_next = trace[(RATE, i+1)]
        #(h_cur, u) = new_rates_inverse([h_prev, h_next, cp_deleted, cp_prev, cp_next])
        #J = jacobian(new_rates_inverse, [h_prev, h_next, cp_deleted, cp_prev, cp_next])[:,1:2]
        #bwd_choices[U] = u
#
        ## set cur rate
        #constraints[(RATE, i)] = h_cur
#
        ## shift down rates
        #for j=i+1:k
            #constraints[(RATE, j)] = trace[(RATE, j+1)]
        #end
    #end
#
    #(new_trace, weight, _, _) = update(trace, model_args, (NoChange(),), constraints)
    #(new_trace, bwd_choices, weight + log(abs(det(J))))
#end

birth_death_move(trace) = metropolis_hastings(trace, birth_death_proposal, (), birth_death_involution, check=true)

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
    cps = [trace[(CHANGEPT, i)] for i=1:k]
    hs = [trace[(RATE, i)] for i=1:k+1]
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
