@testset "enumerative inference" begin

    # polynomial regression model
    @gen function poly_model(n::Int, xs)
        degree ~ uniform_discrete(1, n)
        coeffs = zeros(n+1)
        for d in 0:n
            coeffs[d+1] = {(:coeff, d)} ~ uniform(-1, 1)
        end
        ys = zeros(length(xs))
        for (i, x) in enumerate(xs)
            x_powers = x .^ (0:n)
            y_mean = sum(coeffs[d+1] * x_powers[d+1] for d in 0:degree)
            ys[i] = {(:y, i)} ~ normal(y_mean, 0.1)
        end
        return ys
    end

    # synthetic dataset
    coeffs = [0.5, 0.1, -0.5]
    xs = collect(0.5:0.5:3.0)
    ys = [(coeffs' * [x .^ d for d in 0:2]) for x in xs] 

    observations = choicemap()
    for (i, y) in enumerate(ys)
        observations[(:y, i)] = y
    end

    # test construction of choicemap-volume grid
    grid = choice_vol_grid(
        (:degree, 1:2),
        ((:coeff, 0), -1:0.2:1, :continuous),
        ((:coeff, 1), -1:0.2:1, :continuous),
        ((:coeff, 2), -1:0.2:1, :continuous),
        anchor = :midpoint
    )

    @test size(grid) == (2, 10, 10, 10)
    @test length(grid) == 2000

    choices, log_vol = first(grid)
    @test choices == choicemap(
        (:degree, 1), 
        ((:coeff, 0), -0.9), ((:coeff, 1), -0.9), ((:coeff, 2), -0.9),
    )
    @test log_vol ≈ log(0.2^3)

    test_choices(n::Int, cs) =
        cs[:degree] in 1:n && all(-1.0 <= cs[(:coeff, d)] <= 1.0 for d in 1:n)

    @test all(test_choices(2, choices) for (choices, _) in grid)
    @test all(log_vol ≈ log(0.2^3) for (_, log_vol) in grid)

    # run enumerative inference over grid
    traces, log_norm_weights, lml_est =
        enumerative_inference(poly_model, (2, xs), observations, grid)

    @test size(traces) == (2, 10, 10, 10)
    @test length(traces) == 2000
    @test all(test_choices(2, tr) for tr in traces)

    # test that log-weights are as expected
    log_joint_weights = [get_score(tr) + log(0.2^3) for tr in traces]
    lml_expected = logsumexp(log_joint_weights)
    @test lml_est ≈ lml_expected
    @test all((jw - lml_expected) ≈ w for (jw, w) in zip(log_joint_weights, log_norm_weights))

    # test that polynomial is most likely quadratic
    degree_probs = sum(exp.(log_norm_weights), dims=(2, 3, 4))
    @test argmax(vec(degree_probs)) == 2

    # test that MAP trace recovers the original coefficients
    map_trace_idx = argmax(log_norm_weights)
    map_trace = traces[map_trace_idx]
    @test map_trace[:degree] == 2
    @test map_trace[(:coeff, 0)] == 0.5
    @test map_trace[(:coeff, 1)] == 0.1
    @test map_trace[(:coeff, 2)] == -0.5

    # 2D mixture of normals
    @gen function mixture_model()
        sign ~ bernoulli(0.5)
        mu = sign ? fill(0.5, 2) : fill(-0.5, 2)
        z ~ broadcasted_normal(mu, ones(2))
    end

    # test construction of grid with 2D random variable
    grid = choice_vol_grid(
        (:sign, [false, true]),
        (:z, (-2.0:0.1:2.0, -2.0:0.1:2.0), :continuous, Val(2)),
        anchor = :left
    )

    @test size(grid) == (2, 40, 40)
    @test length(grid) == 3200

    choices, log_vol = first(grid)
    @test choices == choicemap((:sign, false), (:z, [-2.0, -2.0]))
    @test log_vol ≈ log(0.1^2)

    @test all(all([-2.0, -2.0] .<= choices[:z] .<= [2.0, 2.0]) for (choices, _) in grid)
    @test all(log_vol ≈ log(0.1^2) for (_, log_vol) in grid)

    # run enumerative inference over grid
    traces, log_norm_weights, lml_est =
        enumerative_inference(mixture_model, (), choicemap(), grid)

    @test size(traces) == (2, 40, 40)
    @test length(traces) == 3200
    @test all(all([-2.0, -2.0] .<= tr[:z] .<= [2.0, 2.0]) for tr in traces)

    # test that log-weights are as expected
    function expected_logpdf(tr)
        x, y = tr[:z]
        mu = tr[:sign] ? 0.5 : -0.5
        return log(0.5) + logpdf(normal, x, mu, 1.0) + logpdf(normal, y, mu, 1.0)
    end

    log_joint_weights = [expected_logpdf(tr) + log(0.1^2) for tr in traces]
    lml_expected = logsumexp(log_joint_weights)
    @test lml_est ≈ lml_expected
    @test all((jw - lml_expected) ≈ w for (jw, w) in zip(log_joint_weights, log_norm_weights))

    # test that maximal log-weights are at modes
    max_log_weight = maximum(log_norm_weights)
    max_idxs = findall(log_norm_weights .== max_log_weight)

    max_trace_1 = traces[max_idxs[1]]
    @test max_trace_1[:sign] == false
    @test max_trace_1[:z] == [-0.5, -0.5]

    max_trace_2 = traces[max_idxs[2]]
    @test max_trace_2[:sign] == true
    @test max_trace_2[:z] == [0.5, 0.5]

end
