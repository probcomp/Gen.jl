using PyCall
@pyimport matplotlib.pyplot as plt

using Gen

#import Gen: Distribution, logpdf

# Example from Section 4 of Reversible jump Markov chain Monte Carlo
# computation and Bayesian model determination 

########################
# custom distributions #
########################

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


# piecewise homogenous Poisson process 

# n intervals - n + 1 bounds
# (b_1, b_2]
# (b_2, b_3]
# ..
# (b_n, b_{n+1}]

function compute_total(bounds, rates)
    num_intervals = length(rates)
    if length(bounds) != num_intervals + 1
        error("Number of bounds does not match number of rates")
    end
    total = 0.
    bounds_ascending = true
    for i=1:num_intervals
        lower = bounds[i]
        upper = bounds[i+1]
        rate = rates[i]
        len = upper - lower
        if len <= 0
            bounds_ascending = false
        end
        total += len * rate
    end
    (total, bounds_ascending)
end

struct PiecewiseHomogenousPoissonProcess <: Distribution{Vector{Float64}} end
const piecewise_poisson_process = PiecewiseHomogenousPoissonProcess()

function Gen.logpdf(::PiecewiseHomogenousPoissonProcess, x::Vector{Float64}, bounds::Vector{Float64}, rates::Vector{Float64})
    cur = 1
    upper = bounds[cur+1]
    lpdf = 0.
    for xi in sort(x)
        if xi < bounds[1] || xi > bounds[end]
            error("x ($xi) lies outside of interval")
        end
        while xi > upper 
            cur += 1
            upper = bounds[cur+1]
        end
        lpdf += log(rates[cur])
    end
    (total, bounds_ascending) = compute_total(bounds, rates)
    if bounds_ascending
        lpdf - total
    else
        -Inf
    end
end

function Gen.random(::PiecewiseHomogenousPoissonProcess, bounds::Vector{Float64}, rates::Vector{Float64})
    x = Vector{Float64}()
    num_intervals = length(rates)
    for i=1:num_intervals
        lower = bounds[i]
        upper = bounds[i+1]
        rate = (upper - lower) * rates[i]
        n = random(poisson, rate)
        for j=1:n
            push!(x, random(uniform_continuous, lower, upper))
        end
    end
    x
end


#########
# model #
#########

@gen function model(T::Float64)

    # prior on number of change points
    k = @addr(poisson(3.), "k")

    # prior on the location of (sorted) change points
    change_pts = Vector{Float64}(undef, k)
    lower = 0.
    for i=1:k
        cp = @addr(min_uniform_continuous(lower, T, k-i+1), "cp$i")
        change_pts[i] = cp
        lower = cp
    end

    # k + 1 rate values
    # h$i is the rate for cp$(i-1) to cp$i where cp0 := 0 and where cp$(k+1) := T
    alpha = 1.
    beta = 200.
    rates = Float64[@addr(Gen.gamma(alpha, 1. / beta), "h$i") for i=1:k+1]

    # poisson process
    bounds = vcat([0.], change_pts, [T])
    @addr(piecewise_poisson_process(bounds, rates), "points")
end

function render(trace; ymax=0.02)
    T = get_call_record(trace).args[1]
    assignment = get_assignment(trace)
    k = assignment["k"]
    bounds = vcat([0.], sort([assignment["cp$i"] for i=1:k]), [T])
    rates = [assignment["h$i"] for i=1:k+1]
    for i=1:length(rates)
        lower = bounds[i]
        upper = bounds[i+1]
        rate = rates[i]
        plt.plot([lower, upper], [rate, rate], color="black", linewidth=2)
    end
    points = assignment["points"]
    plt.scatter(points, -rand(length(points)) * (ymax/5.), color="black", s=5)
    ax = plt.gca()
    xlim = [0., T]
    plt.plot(xlim, [0., 0.], "--")
    ax[:set_xlim](xlim)
    ax[:set_ylim](-ymax/5., ymax)
end

function show_prior_samples()
    plt.figure(figsize=(16,16))
    T = 40000.
    for i=1:16
        plt.subplot(4, 4, i)
        trace = simulate(model, (T,))
        render(trace; ymax=0.015)
    end
    plt.tight_layout(pad=0)
    plt.savefig("prior_samples.pdf")
end

#############################
# height and position moves #
#############################

@gen function height_proposal(prev, i::Int)
    prev_assignment = get_assignment(prev)
    height = prev_assignment["h$i"]
    @addr(uniform_continuous(height/2., height*2.), "h$i")
end

@gen function position_proposal(prev, i::Int)
    prev_assignment = get_assignment(prev)
    k = prev_assignment["k"]
    lower = (i == 1) ? 0. : prev_assignment["cp$(i-1)"]
    upper = (i == k) ? T : prev_assignment["cp$(i+1)"]
    @addr(uniform_continuous(lower, upper), "cp$i")
end

function height_move(trace)
    k = get_assignment(trace)["k"]
    i = random(uniform_discrete, 1, k+1)
    mh(model, height_proposal, (i,), trace)
end

function position_move(trace)
    k = get_assignment(trace)["k"]
    i = random(uniform_discrete, 1, k)
    mh(model, position_proposal, (i,), trace)
end


######################
# birth / death move #
######################

# insert a new change point at i, where 1 <= i <= k+1
# the current change point at i, and all after, will be shifted right
# the new change point will be placed between cp$(i-1) and the current cp$i
@gen function birth_proposal(prev, T, i::Int)
    prev_assignment = get_assignment(prev)
    k = prev_assignment["k"]
    lower = (i == 1) ? 0. : prev_assignment["cp$(i-1)"]
    upper = (i == k+1) ? T : prev_assignment["cp$i"]
    @addr(uniform_continuous(lower, upper), "new-cp")
    @addr(uniform_continuous(0., 1.), "u")
end

@gen function death_proposal(prev) end

function birth_move_new_heights(cur_height, new_cp, prev_cp, next_cp, u)
    d_prev = new_cp - prev_cp
    d_next = next_cp - new_cp
    @assert d_prev > 0
    @assert d_next > 0
    d_total = d_prev + d_next
    log_cur_height = log(cur_height)
    log_ratio = log(1 - u) - log(u)
    new_h_prev = exp(log_cur_height - (d_prev / d_total) * log_ratio)
    new_h_next = exp(log_cur_height + (d_next / d_total) * log_ratio)
    @assert new_h_prev > 0.
    @assert new_h_next > 0.
    (new_h_prev, new_h_next)
end

const MODEL = :model
const PROPOSAL = :proposal

@inj function birth_injection(T, i::Int)

    # increment k
    k = @read(MODEL => "k")
    @write(k+1, MODEL => "k")

    # changepoints
    for j=1:i-1
        @copy(MODEL => "cp$j", MODEL => "cp$j")
    end
    @copy(PROPOSAL => "new-cp", MODEL => "cp$i")
    for j=i:k
        @copy(MODEL => "cp$j", MODEL => "cp$(j+1)")
    end

    # compute new heights
    cur_height = @read(MODEL => "h$i")
    prev_cp = (i == 1) ? 0. : @read(MODEL => "cp$(i-1)")
    next_cp = (i == k+1) ? T : @read(MODEL => "cp$i")
    new_cp = @read(PROPOSAL => "new-cp")
    u = @read(PROPOSAL => "u")
    (new_h_prev, new_h_next) = birth_move_new_heights(cur_height, new_cp, prev_cp, next_cp, u)

    # heights
    for j=1:i-1
        @copy(MODEL => "h$j", MODEL => "h$j")
    end
    @write(new_h_prev, MODEL => "h$i")
    @write(new_h_next, MODEL => "h$(i+1)")
    for j=i+1:k+1
        @copy(MODEL => "h$j", MODEL => "h$(j+1)")
    end

    @copy(MODEL => "points", MODEL => "points")
end

function death_move_u_new_height(prev_height, next_height, cur_cp, prev_cp, next_cp)
    d_prev = cur_cp - prev_cp
    d_next = next_cp - cur_cp
    @assert d_prev > 0
    @assert d_next > 0
    d_total = d_prev + d_next
    log_prev_height = log(prev_height)
    log_next_height = log(next_height)
    new_height = exp((d_prev / d_total) * log_prev_height + (d_next / d_total) * log_next_height)
    u = prev_height / (prev_height + next_height)
    @assert new_height > 0.
    (new_height, u)
end

@inj function death_injection(T, i::Int)

    # decrement k
    k = @read(MODEL => "k")
    @assert k > 0
    @write(k-1, MODEL => "k")

    # change points
    for j=1:i-1
        @copy(MODEL => "cp$j", MODEL => "cp$j")
    end
    for j=i+1:k
        @copy(MODEL => "cp$j", MODEL => "cp$(j-1)")
    end

    # compute new height
    cur_cp = @read(MODEL => "cp$i")
    prev_cp = (i == 1) ? 0. : @read(MODEL => "cp$(i-1)")
    next_cp = (i == k) ? T : @read(MODEL => "cp$(i+1)")
    prev_height = @read(MODEL => "h$i")
    next_height = @read(MODEL => "h$(i+1)")
    (new_height, u) = death_move_u_new_height(prev_height, next_height, cur_cp, prev_cp, next_cp)

    # heights
    for j=1:i-1
        @copy(MODEL => "h$j", MODEL => "h$j")
    end
    @write(new_height, MODEL => "h$i")
    for j=i+2:k+1
        @copy(MODEL => "h$j", MODEL => "h$(j-1)")
    end

    @copy(MODEL => "points", MODEL => "points")

    @copy(MODEL => "cp$i", PROPOSAL => "new-cp")
    @write(u, PROPOSAL => "u")
end

function birth_move(trace)
    # if k > 0, then prob_b = 0.25 and prob_d = 0.25
    # the probability that the one we introduce will get deleted, given that
    # we choose a death move, is: 1/(k+1); which is also the probability that
    # we choose the i that we choose here
    k = get_assignment(trace)["k"]
    i = random(uniform_discrete, 1, k+1)
    # prob_b = 1, but prob_d = 0.25, so we correct by -log(4)
    correction = (new_trace) -> (k == 0 ? -log(4) : 0)
    T = get_call_record(trace).args[1]
    rjmcmc(model,
        birth_proposal, (T, i),
        death_proposal, (),
        birth_injection, (T, i),
        trace, correction)
end

function death_move(trace)
    k = get_assignment(trace)["k"]
    @assert k > 0
    i = random(uniform_discrete, 1, k)
    correction = (new_trace) -> (k == 0 ? log(4) : 0)
    T = get_call_record(trace).args[1]
    rjmcmc(model,
        death_proposal, (),
        birth_proposal, (T, i),
        death_injection, (T, i),
        trace, correction)
end

##########################
# Generic MCMC inference #
##########################

function resimulation_mh(selection, trace)
    model_args = get_call_record(trace).args
    (new_trace, weight) = regenerate(model, model_args, NoChange(), trace, selection)
    if log(rand()) < weight
        # accept
        return new_trace
    else
        # reject
        return trace
    end
end

k_selection = DynamicAddressSet()
push_leaf_node!(k_selection, "k")

function generic_mcmc_step(trace)
    k = get_assignment(trace)["k"]
    if k > 0
        prob_h = 1./3
        prob_p = 1./3
        prob_change_k = 1./3
    else
        prob_h = 0.
        prob_p = 0.
        prob_change_k = 1
    end
    move_type = random(categorical, [prob_h, prob_p, prob_change_k])
    if move_type == 1
        height_move(trace)
    elseif move_type == 2
        position_move(trace)
    else
        resimulation_mh(k_selection, trace)
    end
end


#########################
# RJMCMC MCMC inference #
#########################0

function mcmc_step(trace)
    k = get_assignment(trace)["k"]
    if k > 0
        prob_h = 0.25
        prob_p = 0.25
        prob_b = 0.25
        prob_d = 0.25
    else
        prob_h = 0.
        prob_p = 0.
        prob_b = 1.
        prob_d = 0.
    end
    move_type = random(categorical, [prob_h, prob_p, prob_b, prob_d])
    if move_type == 1
        height_move(trace)
    elseif move_type == 2
        position_move(trace)
    elseif move_type == 3
        birth_move(trace)
    elseif move_type == 4
        death_move(trace)
    else
        error("Unknown move type $move_type")
    end
end

function do_mcmc(T, num_steps::Int)
    (trace, _) = generate(model, (T,), observations)
    for iter=1:num_steps
        if iter % 1000 == 0
            println("iter $iter of $num_steps, k: $(get_assignment(trace)["k"])")
        end
        #trace = mcmc_step(trace)
        trace = generic_mcmc_step(trace)
    end
    trace
end


########################
# inference experiment #
########################

Gen.load_generated_functions()

import Random
Random.seed!(1)

# load data set
import CSV
function load_data_set()
    df = CSV.read("coal.csv")
    dates = df[1]
    dates = dates .- minimum(dates)
    dates * 365.25 # convert years to days
end

const points = load_data_set()
const T = maximum(points)
const observations = DynamicAssignment()
observations["points"] = points

function show_posterior_samples()
    plt.figure(figsize=(16,16))
    for i=1:16
        println("replicate $i")
        tic()
        plt.subplot(4, 4, i)
        trace = do_mcmc(T, 5000)#10000)
        toc()
        render(trace; ymax=0.015)
    end
    plt.tight_layout(pad=0)
    plt.savefig("posterior_samples.pdf")
end

function get_rate_vector(trace, test_points)
    assignment = get_assignment(trace)
    k = assignment["k"]
    cps = [assignment["cp$i"] for i=1:k]
    hs = [assignment["h$i"] for i=1:k+1]
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
    num_steps = 5000 # 20000
    for reps=1:10
        (trace, _) = generate(model, (T,), observations)
        for iter=1:num_steps
            if iter % 1000 == 0
                println("iter $iter of $num_steps, k: $(get_assignment(trace)["k"])")
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
    plt.figure()
    plt.plot(test_points, posterior_mean_rate, color="black")
    plt.scatter(points, -rand(length(points)) * (ymax/6.), color="black", s=5)
    ax = plt.gca()
    xlim = [0., T]
    plt.plot(xlim, [0., 0.], "--")
    ax[:set_xlim](xlim)
    ax[:set_ylim](-ymax/5., ymax)
    plt.savefig("posterior_mean_rate.pdf")
end

function plot_trace_plot()
    # show the number of clusters
    (trace, _) = generate(model, (T,), observations)
    num_clusters_vec = Int[]
    burn_in = 20000
    for iter=1:burn_in + 5000
        (trace, accept) = mcmc_step(trace)
        if iter > burn_in
            push!(num_clusters_vec, get_assignment(trace)["k"])
        end
    end
    plt.figure()
    plt.plot(num_clusters_vec)
    ax = plt.gca()
    plt.savefig("trace_plot_rjmcmc.pdf")
end

function plot_trace_plot()
    plt.figure(figsize=(8, 4))

    # generic
    (trace, _) = generate(model, (T,), observations)
    num_clusters_vec = Int[]
    burn_in = 20000
    for iter=1:burn_in + 5000
        trace = generic_mcmc_step(trace)
        if iter > burn_in
            push!(num_clusters_vec, get_assignment(trace)["k"])
        end
    end
    plt.subplot(2, 1, 1)
    plt.plot(num_clusters_vec, "r")

    # reversible jump
    (trace, _) = generate(model, (T,), observations)
    height1 = Float64[]
    num_clusters_vec = Int[]
    burn_in = 20000
    for iter=1:burn_in + 5000
        trace = mcmc_step(trace)
        if iter > burn_in
            push!(num_clusters_vec, get_assignment(trace)["k"])
        end
    end
    plt.subplot(2, 1, 2)
    plt.plot(num_clusters_vec, "b")

    ax = plt.gca()
    plt.savefig("trace_plot.pdf")
end



println("showing prior samples...")
show_prior_samples()

println("showing posterior samples...")
show_posterior_samples()

println("estimating posterior mean rate...")
plot_posterior_mean_rate()

println("making trace plot...")
plot_trace_plot()
