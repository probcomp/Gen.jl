@testset "black box variational inference" begin

    Random.seed!(1)

    @gen function model()
        slope = @trace(normal(-1, exp(0.5)), :slope)
        intercept = @trace(normal(1, exp(2.0)), :intercept)
    end

    @gen function approx()
        @param slope_mu::Float64
        @param slope_log_std::Float64
        @param intercept_mu::Float64
        @param intercept_log_std::Float64
        @trace(normal(slope_mu, exp(slope_log_std)), :slope)
        @trace(normal(intercept_mu, exp(intercept_log_std)), :intercept)
    end
    register_parameters!(approx, [:slope_mu, :slope_log_std, :intercept_mu, :intercept_log_std])

    observations = choicemap()
    optimizer = init_optimizer(DecayStepGradientDescent(1., 1000), approx)

    # test regular black box variational inference
    for multithreaded in [false, true]
        init_parameter!((approx, :slope_mu), 0.0)
        init_parameter!((approx, :slope_log_std), 0.0)
        init_parameter!((approx, :intercept_mu), 0.0)
        init_parameter!((approx, :intercept_log_std), 0.0)
        black_box_vi!(model, (), observations, approx, (), optimizer;
            iters=2000, samples_per_iter=100, verbose=false, multithreaded=multithreaded)

        slope_mu = get_parameter_value((approx, :slope_mu))
        slope_log_std = get_parameter_value((approx, :slope_log_std))
        intercept_mu = get_parameter_value((approx, :intercept_mu))
        intercept_log_std = get_parameter_value((approx, :intercept_log_std))
        @test isapprox(slope_mu, -1., atol=0.001)
        @test isapprox(slope_log_std, 0.5, atol=0.001)
        @test isapprox(intercept_mu, 1., atol=0.001)
        @test isapprox(intercept_log_std, 2.0, atol=0.001)
    end

    # smoke test for black box variational inference with Monte Carlo objectives
    for multithreaded in [false, true]
        init_parameter!((approx, :slope_mu), 0.0)
        init_parameter!((approx, :slope_log_std), 0.0)
        init_parameter!((approx, :intercept_mu), 0.0)
        init_parameter!((approx, :intercept_log_std), 0.0)
        black_box_vimco!(model, (), observations, approx, (), optimizer, 20;
            iters=50, samples_per_iter=100, verbose=false, geometric=false,
            multithreaded=multithreaded)
    
        init_parameter!((approx, :slope_mu), 0.0)
        init_parameter!((approx, :slope_log_std), 0.0)
        init_parameter!((approx, :intercept_mu), 0.0)
        init_parameter!((approx, :intercept_log_std), 0.0)
        black_box_vimco!(model, (), observations, approx, (), optimizer, 20;
            iters=50, samples_per_iter=100, verbose=false, geometric=true,
            multithreaded=multithreaded)
    end

end

@testset "minimal vae" begin

    Random.seed!(1)

    n = 10

    @gen function model()
        @param theta::Float64
        for i in 1:n
            z = ({(:z, i)} ~ normal(theta, 1))
            {(:x, i)} ~ normal(z, 1)
        end
    end
    register_parameters!(model, [:theta])

    @gen function approx(xs)
        @param mu_coeffs::Vector{Float64} # 2 x 1; should be [opt_theta / 2, 0.5]
        @param log_std::Float64
        for i in 1:n
            mu = mu_coeffs[1] + mu_coeffs[2] * xs[i]
            {(:z, i)} ~ normal(mu, exp(log_std))
        end
    end
    register_parameters!(approx, [:mu_coeffs, :log_std])

    observations = choicemap()
    xs = Float64[]
    for i in 1:n
        z = normal(-1.0, 1)
        x = normal(z, 1)
        push!(xs, x)
        observations[(:x, i)] = x
    end

    # the optimum maximum marginal likelihood is at theta = mean(xs)
    opt_theta = sum(xs) / length(xs)
    println("opt_theta: $opt_theta")
    # see https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    posterior_precisions = 1.0 + 1 * 1.0 # n = 1
    posterior_means = (xs .+ opt_theta) ./ 2.0
    posterior_precisions = 2.0
    @gen function optimum_approx()
        for i in 1:n
            {(:z, i)} ~ normal(posterior_means[i], sqrt(1.0 / posterior_precisions))
        end
    end
    init_parameter!((model, :theta), opt_theta)
    approx_trace = simulate(optimum_approx, ())
    (model_trace, _) = generate(model, (), merge(get_choices(approx_trace), observations))
    # note that p(z1..zn, x1..xn) / p(z1..zn | x1..xn) = p(x1...xn) - for all z1..zn
    log_marginal_likelihood = get_score(model_trace) - get_score(approx_trace)
    println("true optimum log_marginal_likelihood: $log_marginal_likelihood")

    # using BBVI with score function estimator
    init_parameter!((model, :theta), 0.0)
    init_parameter!((approx, :mu_coeffs), zeros(2))
    init_parameter!((approx, :log_std), 0.0)
    approx_optimizer = init_optimizer(FixedStepGradientDescent(0.0001), approx)
    model_optimizer = init_optimizer(FixedStepGradientDescent(0.002), model)
    @time (_, _, elbo_history, _) =
        black_box_vi!(model, (), model_optimizer, observations,
                      approx, (xs,), approx_optimizer;
                      iters=3000, samples_per_iter=20, verbose=false)
    @test isapprox(get_parameter_value((model, :theta)), opt_theta, atol=1e-1)
    println("final theta: $(get_parameter_value((model, :theta)))")
    println("final elbo estimate: $(elbo_history[end])")
    @test isapprox(elbo_history[end], log_marginal_likelihood, rtol=0.1)

    # using VIMCO
    init_parameter!((model, :theta), 0.0)
    init_parameter!((approx, :mu_coeffs), zeros(2))
    init_parameter!((approx, :log_std), 0.0)
    approx_optimizer = init_optimizer(FixedStepGradientDescent(0.001), approx)
    model_optimizer = init_optimizer(FixedStepGradientDescent(0.01), model)
    @time (_, _, elbo_history, _) =
        black_box_vimco!(model, (), model_optimizer, observations,
                         approx, (xs,), approx_optimizer, 10;
                         iters=1000, samples_per_iter=10, verbose=false)
    println("final theta: $(get_parameter_value((model, :theta)))")
    println("final elbo estimate: $(elbo_history[end])")
    @test isapprox(elbo_history[end], log_marginal_likelihood, rtol=0.1)
end
