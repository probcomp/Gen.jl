using Gen
using PyPlot
import Random

@gen function model()
    if ({:z} ~ bernoulli(0.5))
        m1 = ({:m1} ~ gamma(1, 1))
        m2 = ({:m2} ~ gamma(1, 1))
    else
        m = ({:m} ~ gamma(1, 1))
        (m1, m2) = (m, m)
    end
    {:y1} ~ normal(m1, 0.1)
    {:y2} ~ normal(m2, 0.1)
end

@gen function mean_random_walk_proposal(trace)
    if trace[:z]
        {:m1} ~ normal(trace[:m1], 0.1)
        {:m2} ~ normal(trace[:m2], 0.1)
    else
        {:m} ~ normal(trace[:m], 0.1)
    end
end

@gen function split_merge_proposal(trace)
    if trace[:z]
        # currently two means, switch to one
    else
        # currently one mean, switch to two
        {:u} ~ uniform_continuous(0, 1)
    end
end

function split_mean(m, u)
    log_m = log(m)
    log_ratio = log(1 - u) - log(u)
    m1 = exp(log_m - 0.5 * log_ratio)
    m2 = exp(log_m + 0.5 * log_ratio)
    (m1, m2)
end

function merge_mean(m1, m2)
    log_m1 = log(m1)
    log_m2 = log(m2)
    m = exp(0.5 * log_m1 + 0.5 * log_m2)
    u = m1 / (m1 + m2)
    (m, u)
end

@involution function split_merge_involution(
        model_args, proposal_args, proposal_retval)
    if @read_discrete_from_model(:z)
        # currently two means, switch to one
        @write_discrete_to_model(:z, false)
        m1 = @read_continuous_from_model(:m1)
        m2 = @read_continuous_from_model(:m2)
        (m, u) = merge_mean(m1, m2)
        @write_continuous_to_model(:m, m)
        @write_continuous_to_proposal(:u, u)
    else
        # currently one mean, switch to two
        @write_discrete_to_model(:z, true)
        m = @read_continuous_from_model(:m)
        u = @read_continuous_from_proposal(:u)
        (m1, m2) = split_mean(m, u)
        @write_continuous_to_model(:m1, m1)
        @write_continuous_to_model(:m2, m2)
    end
end

function do_inference_simple(y1, y2)
    trace, = generate(model, (), choicemap((:y1, y1), (:y2, y2), (:z, false), (:m, 1.2)))
    zs = Bool[]
    m = Float64[]
    m1 = Float64[]
    m2 = Float64[]
    for iter=1:200
        trace, = mh(trace, select(:z))
        trace, = mh(trace, mean_random_walk_proposal, ())
        push!(zs, trace[:z])
        push!(m, trace[:z] ? NaN : trace[:m])
        push!(m1, trace[:z] ? trace[:m1] : NaN)
        push!(m2, trace[:z] ? trace[:m2] : NaN)
    end
    (zs, m, m1, m2)
end

function do_inference_rjmcmc(y1, y2)
    trace, = generate(model, (), choicemap((:y1, y1), (:y2, y2), (:z, false), (:m, 1.)))
    zs = Bool[]
    m = Float64[]
    m1 = Float64[]
    m2 = Float64[]
    for iter=1:200
        trace, = mh(
            trace, split_merge_proposal, (), split_merge_involution; check=true)
        trace, = mh(trace, mean_random_walk_proposal, ())
        push!(zs, trace[:z])
        push!(m, trace[:z] ? NaN : trace[:m])
        push!(m1, trace[:z] ? trace[:m1] : NaN)
        push!(m2, trace[:z] ? trace[:m2] : NaN)
    end
    (zs, m, m1, m2)
end

Random.seed!(2)

figure()

y1, y2 = (1.0, 1.3)
(zs, m, m1, m2) = do_inference_rjmcmc(y1, y2)
subplot(2, 2, 1)
plot(zs, label="z")
plot(m, label="m")
plot(m1, label="m1")
plot(m2, label="m2")
title("(y1, y2) = ($y1, $y2), rjmcmc")
legend(loc="lower right")

y1, y2 = (1.0, 1.3)
(zs, m, m1, m2) = do_inference_simple(y1, y2)
subplot(2, 2, 2)
plot(zs, label="z")
plot(m, label="m")
plot(m1, label="m1")
plot(m2, label="m2")
title("(y1, y2) = ($y1, $y2); simple")
legend(loc="lower right")

y1, y2 = (1.0, 3.0)
(zs, m, m1, m2) = do_inference_rjmcmc(y1, y2)
subplot(2, 2, 3)
plot(zs, label="z")
plot(m, label="m")
plot(m1, label="m1")
plot(m2, label="m2")
title("(y1, y2) = ($y1, $y2); rjmcmc")
legend(loc="lower right")

y1, y2 = (1.0, 3.0)
(zs, m, m1, m2) = do_inference_simple(y1, y2)
subplot(2, 2, 4)
plot(zs, label="z")
plot(m, label="m")
plot(m1, label="m1")
plot(m2, label="m2")
title("(y1, y2) = ($y1, $y2); simple")
legend(loc="lower right")

savefig("rjmcmc.png")
