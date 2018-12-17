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

@testset "particle filtering" begin

    prior = [0.2, 0.3, 0.5]

    emission_dists = [
        0.1 0.2 0.7;
        0.2 0.7 0.1;
        0.7 0.2 0.1
    ]'

    transition_dists = [
        0.4 0.4 0.2;
        0.2 0.3 0.5;
        0.9 0.05 0.05
    ]'

    @gen function kernel(t::Int, prev_z::Int, params::Nothing)
        z = @addr(categorical(transition_dists[:,prev_z]), :z)
        @addr(categorical(emission_dists[:,z]), :x)
        return z
    end

    chain = Unfold(kernel)

    @gen function model(num_steps::Int)
        z_init = @addr(categorical(prior), :z_init)
        @addr(categorical(emission_dists[:,z_init]), :x_init)
        @diff argdiff = @argdiff()
        @addr(chain(num_steps-1, z_init, nothing), :chain, argdiff)
    end

    num_steps = 4
    obs_x = [1, 1, 2, 3]

    # latents:
    # :z_init
    # :chain => 1 => :z
    # :chain => 2 => :z
    # :chain => 3 => :z

    # observations :
    # :x_init
    # :chain => 1 => :x
    # :chain => 2 => :x
    # :chain => 3 => :x


@testset "custom proposal" begin

    @gen function init_proposal(x::Int)
        dist = prior .* emission_dists[x,:]
        @addr(categorical(dist ./ sum(dist)), :z_init)
    end

    @gen function step_proposal(t::Int, prev_z::Int, x::Int)
        dist = transition_dists[:,prev_z] .* emission_dists[x,:]
        # NOTE: was missing :chain, this should have been an error..
        @addr(categorical(dist ./ sum(dist)), :chain => t => :z)
    end

    init_observations = DynamicAssignment()
    init_observations[:x_init] = obs_x[1]
    init_proposal_args = (obs_x[1],)

    function step_proposal_args(step::Int, prev_trace)
        @assert step > 1
        @assert isempty(
            get_subassmt(get_assmt(prev_trace), :chain => (step - 1)))
        if step > 2
            prev_z = get_assmt(prev_trace)[:chain => (step - 2) => :z]
        else
            prev_z = get_assmt(prev_trace)[:z_init]
        end
        (step - 1, prev_z, obs_x[step])
    end

    function step_observations(step::Int)
        @assert step > 1
        observations = DynamicAssignment()
        observations[:chain => (step - 1) => :x] = obs_x[step]
        argdiff = UnfoldCustomArgDiff(false, false)
        (observations, argdiff)
    end

    Random.seed!(0)
    num_particles = 10000
    ess_threshold = 10000 # make sure we exercise resampling
    (_, _, log_ml_est) = particle_filter_custom(model, (),
                    num_steps, num_particles, ess_threshold,
                    init_observations, init_proposal_args,
                    step_observations, step_proposal_args,
                    init_proposal, step_proposal; verbose=false)

    expected_log_ml = log(hmm_forward_alg(prior, emission_dists, transition_dists, obs_x))
    @test isapprox(expected_log_ml, log_ml_est, atol=0.01)
end

@testset "default proposal" begin

    init_observations = DynamicAssignment()
    init_observations[:x_init] = obs_x[1]

    function step_observations(step::Int)
        observations = DynamicAssignment()
        observations[:chain => (step - 1) => :x] = obs_x[step]
        argdiff = UnfoldCustomArgDiff(false, false)
        (observations, argdiff)
    end

    Random.seed!(0)
    num_particles = 10000
    ess_threshold = 10000 # make sure we exercise resampling
    (_, _, log_ml_est) = particle_filter_default(model, (),
                    num_steps, num_particles, ess_threshold,
                    init_observations, step_observations; verbose=false)

    expected_log_ml = log(hmm_forward_alg(prior, emission_dists, transition_dists, obs_x))
    @test isapprox(expected_log_ml, log_ml_est, atol=0.02)
end

end
