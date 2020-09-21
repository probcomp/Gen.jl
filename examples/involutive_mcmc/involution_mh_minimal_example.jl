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

@transform split_merge_involution (model_in, aux_in) to (model_out, aux_out) begin
    if @read(model_in[:z], :discrete)
        # currently two means, switch to one
        @write(model_out[:z], false, :discrete)
        m1 = @read(model_in[:m1], :continuous)
        m2 = @read(model_in[:m2], :continuous)
        (m, u) = merge_mean(m1, m2)
        @write(model_out[:m], m, :continuous)
        @write(aux_out[:u], u, :continuous)
    else
        # currently one mean, switch to two
        @write(model_out[:z], true, :discrete)
        m = @read(model_in[:m], :continuous)
        u = @read(aux_in[:u], :continuous)
        (m1, m2) = split_mean(m, u)
        @write(model_out[:m1], m1, :continuous)
        @write(model_out[:m2], m2, :continuous)
    end
end

is_involution!(split_merge_involution)

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

using Plots

function make_plots()
    Random.seed!(2)
    
    y1, y2 = (1.0, 1.3)
    (zs, m, m1, m2) = do_inference_rjmcmc(y1, y2)
    p1 = plot(title="Involution MH (RJMCMC)", m, label="m")
    plot!(m1, label="m1")
    plot!(m2, label="m2")
    ylims!(0.5, 1.5)
    
    p2 = plot(zs, label="z", color="black")
    xlabel!("# MCMC moves")
    yticks!([0, 1], ["F", "T"])
    ylims!(-0.1, 1.1)
    
    y1, y2 = (1.0, 1.3)
    (zs, m, m1, m2) = do_inference_simple(y1, y2)
    p3 = plot(title="Selection MH", m, label="m")
    plot!(m1, label="m1")
    plot!(m2, label="m2")
    ylims!(0.5, 1.5)
    
    p4 = plot(zs, label="z", color="black")
    xlabel!("# MCMC moves")
    yticks!([0, 1], ["F", "T"])
    ylims!(-0.1, 1.1)
    
    plot(p1, p3, p2, p4)
    savefig("rjmcmc.png")
end

make_plots()
