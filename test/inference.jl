# NOTE: these currently just test that the inference methods do not crash
# put longer-running statistical tests of inference methods elsewhere

@testset "importance sampling" begin

    @gen function model()
        x = @addr(normal(0, 1), :x)
        @addr(normal(x, 1), :y)
    end

    @gen function proposal()
        @addr(normal(0, 2), :x)
    end

    y = 2.
    observations = DynamicAssignment()
    set_leaf_node!(observations, :y, y)
    
    n = 4

    (traces, log_weights, lml_est) = importance_sampling(model, (), observations, n)
    @test length(traces) == n
    @test length(log_weights) == n
    @test isapprox(logsumexp(log_weights), 0., atol=1e-14)
    @test !isnan(lml_est)
    for trace in traces
        @test get_assignment(trace)[:y] == y
    end

    (traces, log_weights, lml_est) = importance_sampling(model, (), observations, proposal, (), n)
    @test length(traces) == n
    @test length(log_weights) == n
    @test isapprox(logsumexp(log_weights), 0., atol=1e-14)
    @test !isnan(lml_est)
    for trace in traces
        @test get_assignment(trace)[:y] == y
    end

    (trace, lml_est) = importance_resampling(model, (), observations, n)
    @test isapprox(logsumexp(log_weights), 0., atol=1e-14)
    @test !isnan(lml_est)
    @test get_assignment(trace)[:y] == y

    (trace, lml_est) = importance_resampling(model, (), observations, proposal, (), n)
    @test isapprox(logsumexp(log_weights), 0., atol=1e-14)
    @test !isnan(lml_est)
    @test get_assignment(trace)[:y] == y
end


@testset "SGD training" begin

    Random.seed!(1)

    # x y   z

    # 0 0   0
    # 0 1   1
    # 1 0   1
    # 1 1   0

    # x y   p(x, y | z=0)
    # 0 0   0.5
    # 0 1   0
    # 1 0   0
    # 1 1   0.5

    # x y   p(x, y | z=1)
    # 0 0   0
    # 0 1   0.5
    # 1 0   0.5
    # 1 1   0


    @gen function teacher()
        x = @addr(bernoulli(0.5), :x)
        y = @addr(bernoulli(0.5), :y)
        z::Bool = xor(x, y)
        @addr(bernoulli(z ? 1.0 : 0.0), :z)
    end

    sigmoid(val) = 1.0 / (1.0 + exp(-val))

    @gen function batch_student(zs::Vector{Bool})
        @param theta1::Float64
        @param theta2::Float64
        @param theta3::Float64
        @param theta4::Float64
        @param theta5::Float64
        for (i, z) in enumerate(zs)
            x = @addr(bernoulli(sigmoid(theta1)), i => :x)
            if z && x
                prob_y = sigmoid(theta2) # should be near 0
            elseif z && !x
                prob_y = sigmoid(theta3) # should be near 1
            elseif !z && x
                prob_y = sigmoid(theta4) # should be near 1
            elseif !z && !x
                prob_y = sigmoid(theta5) # should be near 0
            end
            @addr(bernoulli(prob_y), i => :y)
        end
    end

    # theta1 = 0.0
    # theta2 -> inf (prob_y -> 1)
    # theta3 -> -inf (prob_y -> 0)
    
    input_extractor = (samples::Vector) -> (Bool[s[:z] for s in samples],)

    function constraint_extractor(samples::Vector)
        constraints = DynamicAssignment()
        for (i, s) in enumerate(samples)
            constraints[i => :x] = s[:x]
            constraints[i => :y] = s[:y]
        end
        constraints
    end

    function minibatch_callback(batch, minibatch, avg_score, verbose)
        for name in [:theta1, :theta2, :theta3, :theta4, :theta5]
            grad = get_param_grad(batch_student, name)
            value = get_param(batch_student, name)
            set_param!(batch_student, name, value + grad * 0.01)
            zero_param_grad!(batch_student, name)
        end
    end

    batch_callback = (batch, verbose) -> nothing

    init_param!(batch_student, :theta1, 0.)
    init_param!(batch_student, :theta2, 0.)
    init_param!(batch_student, :theta3, 0.)
    init_param!(batch_student, :theta4, 0.)
    init_param!(batch_student, :theta5, 0.)

    # check gradients using finite differences on a simulated batch
    assignments = Vector{Any}(undef, 100)
    for i=1:100
        assignments[i] = get_assignment(simulate(teacher, ()))
    end
    input = input_extractor(assignments)
    constraints = constraint_extractor(assignments)
    student_trace = assess(batch_student, input, constraints)
    backprop_params(batch_student, student_trace, nothing)
    for name in [:theta1, :theta2, :theta3, :theta4, :theta5]
        actual = get_param_grad(batch_student, name)
        dx = 1e-6
        value = get_param(batch_student, name)
        set_param!(batch_student, name, value + dx)
        lpdf_pos = get_call_record(assess(batch_student, input, constraints)).score
        set_param!(batch_student, name, value - dx)
        lpdf_neg = get_call_record(assess(batch_student, input, constraints)).score
        set_param!(batch_student, name, value)
        expected = (lpdf_pos - lpdf_neg) / (2 * dx)
        @test isapprox(actual, expected, atol=1e-4)
    end

    conf = SGDTrainConf(2000, 50, 1, 50,
        input_extractor, constraint_extractor,
        minibatch_callback, batch_callback)
    sgd_train_batch(teacher, (), batch_student, conf, false)

    # p(x | z=0) = p(x | z=1) = 0.5
    @test isapprox(get_param(batch_student, :theta1), 0., atol=0.2)

    # y | z, x = xor(x, z)
    @test get_param(batch_student, :theta2) < -5
    @test get_param(batch_student, :theta3) > 5
    @test get_param(batch_student, :theta4) > 5
    @test get_param(batch_student, :theta5) < -5
end

function hmm_forward_alg(prior::Vector{Float64},
        emission_dists::AbstractArray{Float64,2}, transition_dists::AbstractArray{Float64,2},
        emissions::Vector{Int})
    marg_lik = 1.
    alpha = prior # p(z_1)
    for i=2:length(emissions)

        # p(z_{i-1} , y_{i-1} | y_{1:i-2}) for each z_{i-1}
        prev_posterior = alpha .* emission_dists[emissions[i-1], :] 

        # p(y_{i-1} | y_{1:i-2})
        denom = sum(prev_posterior) 

        # p(z_{i-1} | y_{1:i-1})
        prev_posterior = prev_posterior / denom 

        # p(z_i | y_{1:i-1})
        alpha = transition_dists * prev_posterior

        # p(y_{1:i-1})
        marg_lik *= denom
    end
    prev_posterior = alpha .* emission_dists[emissions[end], :] 
    denom = sum(prev_posterior) 
    marg_lik *= denom
    marg_lik
end

@testset "hmm forward alg" begin

    # test the hmm_forward_alg on a hand-calculated example
    prior = [0.4, 0.6]
    emission_dists = [0.1 0.9; 0.7 0.3]'
    transition_dists = [0.5 0.5; 0.2 0.8']
    obs = [2, 1]
    expected_marg_lik = 0.
    # z = [1, 1]
    expected_marg_lik += prior[1] * transition_dists[1, 1] * emission_dists[obs[1], 1] * emission_dists[obs[2], 1]
    # z = [1, 2]
    expected_marg_lik += prior[1] * transition_dists[2, 1] * emission_dists[obs[1], 1] * emission_dists[obs[2], 2]
    # z = [2, 1]
    expected_marg_lik += prior[2] * transition_dists[1, 2] * emission_dists[obs[1], 2] * emission_dists[obs[2], 1]
    # z = [2, 2]
    expected_marg_lik += prior[2] * transition_dists[2, 2] * emission_dists[obs[1], 2] * emission_dists[obs[2], 2]
    actual_marg_lik = hmm_forward_alg(prior, emission_dists, transition_dists, obs)
    @test isapprox(actual_marg_lik, expected_marg_lik)
    
end

@testset "particle filtering with custom proposal" begin

    prior = [0.2, 0.3, 0.5]

    emission_dists = [
        0.1 0.2 0.8;
        0.2 0.8 0.1;
        0.8 0.2 0.1
    ]'

    transition_dists = [
        0.4 0.4 0.2;
        0.2 0.3 0.5;
        0.9 0.05 0.05
    ]'

    @gen function model_init()
        z = @addr(categorical(prior), :z)
        @addr(categorical(emission_dists[:,z]), :x)
        return z
    end

    @gen function model_step(t::Int, prev_z::Int, params::Nothing)
        z = @addr(categorical(transition_dists[:,prev_z]), :z)
        @addr(categorical(emission_dists[:,z]), :x)
        return z
    end

    @gen function model(num_steps::Int)
        z_init = @addr(model_init(), :init)
        change = UnfoldCustomArgDiff(true, false, false) # we are only ever extending
        @addr(Unfold(model_step)(num_steps-1, z_init, nothing), :unfold, change)
    end

    num_steps = 4
    obs_x = [1, 1, 2, 3]

    # latents:
    # :init => z
    # :unfold => 1 => :z
    # :unfold => 2 => :z
    # :unfold => 3 => :z

    # observations :
    # :init => :x
    # :unfold => 1 => :x
    # :unfold => 2 => :x
    # :unfold => 3 => :x

    @gen function proposal_init(x::Int)
        dist = prior .* emission_dists[x,:]
        @addr(categorical(dist ./ sum(dist)), :init => :z)
    end

    @gen function proposal_step(t::Int, prev_z::Int, x::Int)
        dist = transition_dists[:,prev_z] .* emission_dists[x,:]
        # NOTE: was missing :unfold, this should have been an error..
        @addr(categorical(dist ./ sum(dist)), :unfold => t => :z)
    end

    function get_init_observations_and_proposal_args()
        observations = DynamicAssignment()
        observations[:init => :x] = obs_x[1]
        init_proposal_args = (obs_x[1],)
        (observations, init_proposal_args)
    end

    function get_step_observations_and_proposal_args(step::Int, prev_trace)
        @assert !has_internal_node(get_assignment(prev_trace), :unfold => (step - 1))
        @assert step > 1
        if step == 2
            prev_z = get_assignment(prev_trace)[:init => :z]
        else
            prev_z = get_assignment(prev_trace)[:unfold => (step - 2) => :z]
        end
        observations = DynamicAssignment()
        observations[:unfold => (step - 1) => :x] = obs_x[step]
        step_proposal_args = (step - 1, prev_z, obs_x[step])
        (observations, step_proposal_args, unknownargdiff)
    end
    Random.seed!(0)

    num_particles = 10000
    ess_threshold = 10000 # make sure we exercise resampling
    (_, _, log_ml_est) = particle_filter(model, (),
                    num_steps, num_particles, ess_threshold,
                    get_init_observations_and_proposal_args,
                    get_step_observations_and_proposal_args,
                    proposal_init, proposal_step; verbose=true)

    expected_log_ml = log(hmm_forward_alg(prior, emission_dists, transition_dists, obs_x))
    println("expected: $expected_log_ml")
    println("expected: $log_ml_est")
    @test isapprox(expected_log_ml, log_ml_est, atol=0.01)
end

@testset "particle filtering with internal proposal" begin

    prior = [0.2, 0.3, 0.5]

    emission_dists = [
        0.1 0.2 0.8;
        0.2 0.8 0.1;
        0.8 0.2 0.1
    ]'

    transition_dists = [
        0.4 0.4 0.2;
        0.2 0.3 0.5;
        0.9 0.05 0.05
    ]'

    @gen function model_init()
        z = @addr(categorical(prior), :z)
        @addr(categorical(emission_dists[:,z]), :x)
        return z
    end

    @gen function model_step(t::Int, prev_z::Int, params::Nothing)
        z = @addr(categorical(transition_dists[:,prev_z]), :z)
        @addr(categorical(emission_dists[:,z]), :x)
        return z
    end

    @gen function model(num_steps::Int)
        z_init = @addr(model_init(), :init)
        change = UnfoldCustomArgDiff(true, false, false) # we are only ever extending
        @addr(Unfold(model_step)(num_steps-1, z_init, nothing), :unfold, change)
    end

    num_steps = 4
    obs_x = [1, 1, 2, 3]

    # latents:
    # :init => z
    # :unfold => 1 => :z
    # :unfold => 2 => :z
    # :unfold => 3 => :z

    # observations :
    # :init => :x
    # :unfold => 1 => :x
    # :unfold => 2 => :x
    # :unfold => 3 => :x

    function get_observations(step::Int)
        observations = DynamicAssignment()
        if step == 1
            observations[:init => :x] = obs_x[step]
        else
            observations[:unfold => (step-1) => :x] = obs_x[step]
        end
        return (observations, unknownargdiff)
    end

    Random.seed!(0)

    num_particles = 10000
    ess_threshold = 10000 # make sure we exercise resampling
    (_, _, log_ml_est) = particle_filter(model, (),
                    num_steps, num_particles, ess_threshold,
                    get_observations; verbose=true)

    expected_log_ml = log(hmm_forward_alg(prior, emission_dists, transition_dists, obs_x))
    println("expected: $expected_log_ml")
    println("actual: $log_ml_est")
    @test isapprox(expected_log_ml, log_ml_est, atol=0.02)
end
