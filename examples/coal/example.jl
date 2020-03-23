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
    alpha, beta = 1., 200.
    rates = Float64[({(:rate, i)} ~ Gen.gamma(alpha, 1. / beta)) for i=1:k+1]

    # poisson process
    if k == 0
        bounds = [0., 1.]
    else
        bounds = [0., 0.5, 1.]
    end
    events ~ piecewise_poisson_process(bounds, rates)
end

@gen function proposal(trace)
    k = trace[:k]
    if k == 0
        u ~ uniform_continuous(0, 1)
    end
end

function discrete_involution(trace, u, proposal_args)
    k = trace[:k]
    constraints = choicemap((:k, k == 0 ? 1 : 0))
    u_back = choicemap()
    retval = k
    (constraints, u_back, retval)
end

expr = MacroTools.prewalk(MacroTools.rmlines, macroexpand(Main, :(
@involution function continuous_involution(model_args, proposal_args, f_disc_retval)

    k = f_disc_retval

    if k == 0

        rate = @read_from_model((:rate, 1))
        u = @read_from_proposal(:u)

        prev_rate, next_rate = split_rates(rate, u)

        @write_to_model((:rate, 1), prev_rate)
        @write_to_model((:rate, 2), next_rate)

    else

        prev_rate = @read_from_model((:rate, 1))
        next_rate = @read_from_model((:rate, 2))

        rate, u = merge_rate(prev_rate, next_rate)

        @write_to_model((:rate, 1), rate)
        @write_to_proposal(:u, u)
    end

end
)))

println(expr)

eval(expr)

### experiment 

function do_experiment()
    events = [0.4, 0.2, 0.3, 0.4, 0.3, 0.2, 0.8]
    obs = choicemap((:events, events))
    (trace, _) = generate(model, (), obs)
    for iter=1:100
        trace, acc = rjmcmc(
            trace, proposal, (), discrete_involution, continuous_involution)
        println("acc: $acc")
    end
end

do_experiment()





