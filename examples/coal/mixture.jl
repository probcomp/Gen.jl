using PyCall
@pyimport matplotlib.pyplot as plt
using Gen


#######################################
# mixture of two normal distributions #
#######################################

import Distributions
import Gen: random, logpdf
struct TwoNormals <: Distribution{Float64} end
const two_normals = TwoNormals()

function logpdf(::TwoNormals, x, w1, mu1, mu2, sigma1, sigma2)
    if sigma1 < 0 || sigma2 < 0
        return -Inf
    end
    l1 = Distributions.logpdf(Distributions.Normal(mu1, sigma1), x) + log(w1)
    l2 = Distributions.logpdf(Distributions.Normal(mu2, sigma2), x) + log(1 - w1)
    m = max(l1, l2)
    m + log(exp(l1 - m) + exp(l2 - m))
end

function logpdf_grad(::TwoNormals, x, w1, mu1, m2, sigma1, sigma2)
    l1 = Distributions.logpdf(Distributions.Normal(mu1, sigma1), x) + log(w1)
    l2 = Distributions.logpdf(Distributions.Normal(mu2, sigma2), x) + log(1 - w1)
    (deriv_x_1, deriv_mu_1, deriv_sigma_1) = logpdf_grad(normal, x, mu1, sigma1)
    (deriv_x_2, deriv_mu_2, deriv_sigma_2) = logpdf_grad(normal, x, mu2, sigma2)
    w1 = 1.0 / (1.0 + exp(l1 - l2))
    w2 = 1.0 / (1.0 + exp(l2 - l1))
    @assert isapprox(w1 + w2, 1.0)
    deriv_x = deriv_x_1 * w1 + deriv_x_2 * w2
    (deriv_x, NaN, w1 * deriv_mu_1, w2 * deriv_mu_2, w1 * deriv_std_1, w2 * deriv_std_2)
end

function random(::TwoNormals, w1, mu1, mu2, sigma1, sigma2)
    if rand() < w1
        mu1 + sigma1 * randn()
    else
        mu2 + sigma2 * randn()
    end
end

has_output_grad(::TwoNormals) = true
has_argument_grads(::TwoNormals) = (false, true, true, true, true)

#########
# model #
#########

@gen function model(n::Int)

    # model selection

    if @trace(bernoulli(0.5), :branch)

        # one cluster
        mu = @trace(normal(0, 100), :mu)
        std = @trace(gamma(1, 1), :std)
        for i=1:n
            @trace(normal(mu, std), "y-$i")
        end

    else

        # two clusters
        w1 = @trace(beta(1, 1), :w1)
        mu1 = @trace(normal(0, 100), :mu1)
        mu2 = @trace(normal(0, 100), :mu2)
        std1 = @trace(gamma(1, 1), :std1)
        std2 = @trace(gamma(1, 1), :std2)
        for i=1:n
            @trace(two_normals(w1, mu1, mu2, std1, std2), "y-$i")
        end

    end
end


##################################
# generic trans-dimensional move #
##################################

branch_selection = select(:branch)

function generic_transdim_move(trace)
    resimulation_mh(trace, branch_selection)
end

one_cluster_params = select(:mu, :std)

two_cluster_params = select(:mu1, :mu2, :std1, :std2)

w1_selection = select(:w1)

function fixed_dim_move(trace)
    if get_choices(trace)[:branch]
        (trace, _) = default_mh(one_cluster_params, trace)
        (trace, _) = mala(trace, one_cluster_params, 0.01)
    else
        (trace, _) = default_mh(two_cluster_params, trace)
        (trace, _) = mala(trace, two_cluster_params, 0.01)
    end
    trace
end


###############
# RJMCMC move #
###############

@gen function split_proposal(prev)
    u1 = @trace(beta(2, 2), :u1)
    u2 = @trace(beta(2, 2), :u2)
    u3 = @trace(beta(1, 1), :u3)
end

@gen function merge_proposal(prev) end

const MODEL = :model
const PROPOSAL = :proposal

@inj function split_injection(n::Int)
    @assert @read(:model => :branch)
    @write(false, :model => :branch)
    mu = @read(:model => :mu)
    std = @read(:model => :std)
    u1 = @read(:proposal => :u1)
    u2 = @read(:proposal => :u2)
    u3 = @read(:proposal => :u3)
    w1 = u1
    w2 = 1.0 - u1

    # always make mu1 < mu2
    mu1 = mu - u2 * std * sqrt(w2 / w1)
    mu2 = mu + u2 * std * sqrt(w1 / w2)

    var = std * std
    std1 = sqrt(u3 * (1 - u2 * u2) * var / w1)
    std2 = sqrt((1 - u3) * (1 - u2 * u2) * var / w2)

    @write(mu1, :model => :mu1)
    @write(std1, :model => :std1)
    @write(mu2, :model => :mu2)
    @write(std2, :model => :std2)
    @write(w1, :model => :w1)
    for i=1:n
        @copy(:model => "y-$i", :model => "y-$i")
    end
end

@inj function merge_injection(n::Int)
    @assert !@read(:model => :branch)
    @write(true, :model => :branch)
    w1 = @read(:model => :w1)
    w2 = 1 - w1
    mu1 = @read(:model => :mu1)
    mu2 = @read(:model => :mu2)
    std1 = @read(:model => :std1)
    std2 = @read(:model => :std2)
    u1 = w1

    # make mu1 <= mu2
    if mu1 > mu2
        tmp = mu1
        mu1 = mu2
        mu2 = tmp
    end

    @write(u1, :proposal => :u1)
    mu = w1 * mu1 + w2 * mu2
    @write(mu, :model => :mu)
    var = w1 * (mu1 * mu1 + std1 * std1) + w2 * (mu2 * mu2 + std2 * std2) - mu * mu
    std = sqrt(var)
    @write(std, :model => :std)
    u2 = (mu - mu1) / (std * sqrt(w2 / w1))
    std_ratio = std1 / std2
    weighted_var_ratio = std_ratio * std_ratio * w1 / w2
    u3 = weighted_var_ratio / (1 + weighted_var_ratio)
    @write(u2, :proposal => :u2)
    @write(u3, :proposal => :u3)
    for i=1:n
        @copy(:model => "y-$i", :model => "y-$i")
    end
end

correction = (new_trace) -> 0.
function general_mh_transdim_move(trace)
    branch = get_choices(trace)[:branch]
    n = get_call_record(trace).args[1]
    if branch
        general_mh(model,
            split_proposal, (),
            merge_proposal, (),
            split_injection, (n,),
            trace, correction)
    else
        general_mh(model,
            merge_proposal, (),
            split_proposal, (),
            merge_injection, (n,),
            trace, correction)
    end
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
    ys = df[1]
    ys = ys - mean(ys)
    ys = ys ./ std(ys)
    return ys
end

#const ys = load_data_set()
ys = Float64[-1.0, 1.0]
const n = length(ys)

function plot_trace_plot()
    plt.figure(figsize=(6, 5))

    observations = choicemap()
    for (i, y) in enumerate(ys)
        observations["y-$i"] = y
    end

    # RJMCMC
    (trace, _) = generate(model, (n,), observations)
    model_choice_vec = Bool[]
    burn_in = 10000
    scores = Float64[]
    for iter=1:burn_in + 2000
        (trace, accept) = general_mh_transdim_move(trace)
        trace = fixed_dim_move(trace)
        score = get_call_record(trace).score
        if iter > burn_in
            push!(model_choice_vec, get_choices(trace)[:branch])
            push!(scores, score)
        end
        println("iter: $iter, score: $score")
    end
    plt.subplot(2, 1, 1)
    plt.plot(convert(Vector{Int}, model_choice_vec) .+ 1, "r")
    plt.ylabel("Selected Model")
    plt.title("Custom Reversible Jump Trans-Dimensional Moves")

    # generic
    (trace, _) = generate(model, (n,), observations)
    model_choice_vec = Bool[]
    burn_in = 10000
    scores = Float64[]
    for iter=1:burn_in + 2000
        (trace, accept) = generic_transdim_move(trace)
        trace = fixed_dim_move(trace)
        score = get_call_record(trace).score
        if iter > burn_in
            push!(model_choice_vec, get_choices(trace)[:branch])
            push!(scores, score)
        end
        println("iter: $iter, score: $score")
    end
    plt.subplot(2, 1, 2)
    plt.plot(convert(Vector{Int}, model_choice_vec) .+ 1, "r")
    plt.ylabel("Selected Model")
    plt.title("Generic Trans-Dimensional Moves")

    ax = plt.gca()
    plt.tight_layout()
    plt.savefig("mixture_trace_plot.png")
end

plot_trace_plot()
