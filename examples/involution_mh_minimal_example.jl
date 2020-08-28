using Gen
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

function merge_mean(m1, m2)
    m = sqrt(m1 * m2)
    u = m1 / (m1 + m2)
    (m, u)
end

function split_mean(m, u)
    m1 = m * sqrt((u / (1 - u)))
    m2 = m * sqrt(((1 - u) / u))
    (m1, m2)
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
    for iter=1:100
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
    trace, = generate(model, (), choicemap((:y1, y1), (:y2, y2), (:z, false), (:m, 1.2)))
    zs = Bool[]
    m = Float64[]
    m1 = Float64[]
    m2 = Float64[]
    for iter=1:100
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

y1, y2 = (1.0, 1.3)
(zs, m, m1, m2) = do_inference_rjmcmc(y1, y2)
println(zs)

function plots()
    Random.seed!(2)

    figure(figsize=(6, 3))

    subplot(2, 2, 1)
    y1, y2 = (1.0, 1.3)
    (zs, m, m1, m2) = do_inference_rjmcmc(y1, y2)
    plot(m, label="m")
    plot(m1, label="m1")
    plot(m2, label="m2")
    title("Involution MH (RJMCMC)")
    legend(loc="lower right")
    gca().set_ylim(0.5, 1.5)

    subplot(2, 2, 3)
    plot(zs, label="z", color="black")
    xlabel("\\# MCMC moves")
    legend(loc="center right")
    yticks(ticks=[0, 1], labels=["F", "T"])
    gca().set_ylim(-0.1, 1.1)

    subplot(2, 2, 2)
    y1, y2 = (1.0, 1.3)
    (zs, m, m1, m2) = do_inference_simple(y1, y2)
    plot(m, label="m")
    plot(m1, label="m1")
    plot(m2, label="m2")
    title("Selection MH")
    legend(loc="lower right")
    gca().set_ylim(0.5, 1.5)

    subplot(2, 2, 4)
    plot(zs, label="z", color="black")
    xlabel("\\# MCMC moves")
    legend(loc="center right")
    yticks(ticks=[0, 1], labels=["F", "T"])
    gca().set_ylim(-0.1, 1.1)

    tight_layout()
    savefig("rjmcmc.png")
end

using PyPlot
plots()
