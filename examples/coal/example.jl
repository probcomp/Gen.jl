using Gen

include("rjmcmc.jl")
include("poisson_process.jl")

@gen function model()

    k ~ uniform_discrete(0, 1) # zero or one change point

    # rates
    rates = Float64[]
    for i=1:k+1
        push!(rates, ({(:rate, i)} ~ uniform_continuous(0., 100.)))
    end
    
    ##alpha, beta = 1., 1.
    #rates = Float64[({(:rate, i)} ~ Gen.gamma(alpha, 1. / beta)) for i=1:k+1]

    # poisson process
    if k == 0
        bounds = [0., 1.]
    else
        bounds = [0., 0.5, 1.]
    end
    events ~ piecewise_poisson_process(bounds, rates)
end

#############
# rate move #
#############

@gen function rate_proposal(trace)
    k = trace[:k]

    if k == 0
        segment = 1
    else
        @assert k == 1
        # pick a random segment whose rate to change
        segment ~ uniform_discrete(1, 2)
    end

    # propose new value for the rate
    cur_rate = trace[(:rate, segment)]
    new_rate ~ uniform_continuous(cur_rate/2., cur_rate*2.)

    segment
end

@involution function rate_involution(model_args, proposal_args, proposal_retval::Int)

    segment = proposal_retval

    if @read_discrete_from_model(:k) == 1
        @write_discrete_to_proposal(:segment, segment)
    end

    new_rate = @read_continuous_from_proposal(:new_rate)
    @write_continuous_to_model((:rate, segment), new_rate)
    prev_rate = @read_continuous_from_model((:rate, segment))
    @write_continuous_to_proposal(:new_rate, prev_rate)
end

####################
# split/merge move #
####################

@gen function split_merge_proposal(trace)
    k = trace[:k]
    if k == 0
        # split
        u ~ uniform_continuous(0, 1)
    else
        # merge
    end
    nothing
end

function split_rates(rate, u)
    log_rate = log(rate)
    log_ratio = log(1 - u) - log(u)
    prev_rate = exp(log_rate - 0.5 * log_ratio)
    next_rate = exp(log_rate + 0.5 * log_ratio)
    @assert prev_rate > 0.
    @assert next_rate > 0.
    (prev_rate, next_rate)
end

function merge_rate(prev_rate, next_rate)
    log_prev_rate = log(prev_rate)
    log_next_rate = log(next_rate)
    rate = exp(0.5 * log_prev_rate + 0.5 * log_next_rate)
    u = prev_rate / (prev_rate + next_rate)
    @assert rate > 0.
    (rate, u)
end

@involution function split_merge_involution(model_args, proposal_args, proposal_retval::Nothing)

    k = @read_discrete_from_model(:k)
    @write_discrete_to_model(:k, k == 0 ? 1 : 0)

    if k == 0

        rate = @read_continuous_from_model((:rate, 1))
        u = @read_continuous_from_proposal(:u)

        prev_rate, next_rate = split_rates(rate, u)

        @write_continuous_to_model((:rate, 1), prev_rate)
        @write_continuous_to_model((:rate, 2), next_rate)

    else
        @assert k == 1

        prev_rate = @read_continuous_from_model((:rate, 1))
        next_rate = @read_continuous_from_model((:rate, 2))

        rate, u = merge_rate(prev_rate, next_rate)

        @write_continuous_to_model((:rate, 1), rate)
        @write_continuous_to_proposal(:u, u)
    end
end

##############
# experiment #
##############

function do_experiment()
    events = rand(50) * 0.5
    #events = vcat(events, rand(50) * 0.5 .+ 0.5)
    obs = choicemap((:events, events))
    (trace, _) = generate(model, (), obs)
    @time for iter=1:1000
        trace, acc = metropolis_hastings(
            trace, rate_proposal, (), rate_involution;
            check=true, observations=obs)
        trace, acc = metropolis_hastings(
            trace, split_merge_proposal, (), split_merge_involution;
            check=true, observations=obs)
        println("k: $(trace[:k]), rates: $([trace[(:rate, segment)] for segment in 1:trace[:k]+1])")
    end
end

@time do_experiment()
@time do_experiment()
