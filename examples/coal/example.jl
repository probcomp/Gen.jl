using Gen

include("rjmcmc.jl")
include("poisson_process.jl")

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

function rate_involution_disc(trace, u, proposal_args, proposal_retval::Int)
    #segment = proposal_retval
    #constraints = choicemap()
    #u_back = choicemap()
    #if trace[:k] == 1
        #u_back[:segment] = segment
    #end
    (constraints, u_back, segment) # note: the retval cannot depend on continuous parts of the trace or proposal
end

@involution function rate_involution_cont(model_args, proposal_args, segment::Int)
    new_rate = @read_from_proposal(:new_rate)
    @write_to_model((:rate, segment), new_rate)
    prev_rate = @read_from_model((:rate, segment))
    @write_to_proposal(:new_rate, prev_rate)
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

function split_merge_involution_disc(trace, u, proposal_args, proposal_retval)
    k = trace[:k]
    constraints = choicemap((:k, k == 0 ? 1 : 0))
    u_back = choicemap()
    (constraints, u_back, k)
end

@involution function split_merge_involution_cont(model_args, proposal_args, f_disc_retval)
    k = f_disc_retval

    if k == 0

        rate = @read_from_model((:rate, 1))
        u = @read_from_proposal(:u)

        prev_rate, next_rate = split_rates(rate, u)

        @write_to_model((:rate, 1), prev_rate)
        @write_to_model((:rate, 2), next_rate)

    else
        @assert k == 1

        prev_rate = @read_from_model((:rate, 1))
        next_rate = @read_from_model((:rate, 2))

        rate, u = merge_rate(prev_rate, next_rate)

        @write_to_model((:rate, 1), rate)
        @write_to_proposal(:u, u)
    end
end

##############
# experiment #
##############

function do_experiment()
    events = rand(50) * 0.5
    events = vcat(events, rand(25) * 0.5 .+ 0.5)
    obs = choicemap((:events, events))
    (trace, _) = generate(model, (), obs)
    @time for iter=1:1000
        trace, acc = rjmcmc(
            trace, rate_proposal, (), rate_involution_disc, rate_involution_cont;
            check=true, observations=obs)
        trace, acc = rjmcmc(
            trace, split_merge_proposal, (), split_merge_involution_disc, split_merge_involution_cont;
            check=true, observations=obs)
        println("k: $(trace[:k]), rates: $([trace[(:rate, segment)] for segment in 1:trace[:k]+1])")
    end
end

do_experiment()
