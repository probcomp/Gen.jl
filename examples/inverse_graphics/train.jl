import FileIO

include("model.jl")

function generate_prior_samples()
    for i=1:100
        trace = simulate(model, ())
        img = get_call_record(trace).retval
        img = max.(zeros(img), img)
        img = min.(ones(img), img)
        id = @sprintf("%03d", i)
        FileIO.save("prior/img-$id.png", colorview(Gray, img))
    end
end
println("generating prior samples..")
generate_prior_samples()

const num_train = 100 #10000

function generate_training_data()
    traces = Vector{Any}(num_train)
    for i=1:num_train
        traces[i] = get_assmt(simulate(model, ()))
        if i % 100 == 0
            println("$i of $num_train")
        end
    end
    traces
end

mutable struct ADAMOptimizerState
    m::Dict{Symbol,Any}
    v::Dict{Symbol,Any}
    t::Float64
end

function ADAMOptimizerState()
    m = Dict{Symbol,Any}()
    v = Dict{Symbol,Any}()
    for param in [:W1, :b1, :W2, :b2, :W3, :b3]
        m[param] = zero(proposal.params[param])
        v[param] = zero(proposal.params[param])
    end
    ADAMOptimizerState(m, v, 1.)
end

function update_params!(state::ADAMOptimizerState)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    alpha = 0.00001
    for param in [:W1, :b1, :W2, :b2, :W3, :b3]
        grad = -get_param_grad(proposal, param)
        m_new = beta1 .* state.m[param] + (1-beta1).*grad
        v_new = beta2 .* state.v[param] + (1-beta2).*(grad.^2)
        m_hat = m_new ./ (1. - beta1.^state.t)
        v_hat = v_new ./ (1. - beta2.^state.t)
        current = get_param(proposal, param)
        new = current - alpha * m_hat ./ (sqrt.(v_hat) + eps)
        set_param!(proposal, param, new)
        zero_param_grad!(proposal, param)
        state.m[param] = m_new
        state.v[param] = v_new
    end
    state.t += 1
end

function save_params(gen_func, filename)
    data = Dict(String(name) => param for (name, param) in gen_func.params)
    FileIO.save(filename, data)
end

function train_inference_network(all_traces, num_iter)

    # initialize parameters
    init_param!(proposal, :W1, randn(num_hidden1, num_input) * sqrt(2. / num_input))
    init_param!(proposal, :b1, zeros(num_hidden1))
    init_param!(proposal, :W2, randn(num_hidden2, num_hidden1) * sqrt(2. / num_hidden1))
    init_param!(proposal, :b2, zeros(num_hidden2))
    init_param!(proposal, :W3, randn(num_output, num_hidden2) * sqrt(2. / num_hidden2))
    init_param!(proposal, :b3, zeros(num_output))

    # initialize optimizer
    state = ADAMOptimizerState()

    # do training
    minibatch_size = 100
    tic()
    for iter=1:num_iter
        minibatch = randperm(num_train)[1:minibatch_size]
        traces = all_traces[minibatch]
        @assert length(traces) == minibatch_size
        scores = Vector{Float64}(minibatch_size)
        for (i, model_choices) in enumerate(traces)
            (proposal_trace, _) = project(proposal, (model_choices,), model_choices) # TODO add a function to translate, can't use project() for this..
            backprop_params(proposal, proposal_trace, nothing)
            scores[i] = get_call_record(proposal_trace).score
        end
        update_params!(state)
        println("iter: $iter, score: $(mean(scores))")
        if iter % 10 == 0
            save_params(proposal, "params.jld2")
        end
    end
    toc()
end

println("generating training data...")
tic()
const traces = generate_training_data()
toc()

println("training...")
tic()
train_inference_network(traces, 100)# 10000)
toc()
