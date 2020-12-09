uniform_beta_mixture = @het_mixture Float64 (uniform, 2) (beta, 2)

@testset "fixed mixture of different distributions" begin

    @test !is_discrete(uniform_beta_mixture)
    @test has_output_grad(uniform_beta_mixture)
    @test has_argument_grads(uniform_beta_mixture) == (true, true, true, true, true)

    w1 = 0.4
    w2 = 0.6
    a = 0.0
    b = 3.4
    alpha = 2.3
    beta = 1.0

    # random
    x = uniform_beta_mixture([w1, w2], a, b, alpha, beta)

    # logpdf
    x = 0.123
    actual = logpdf(uniform_beta_mixture, x, [w1, w2], a, b, alpha, beta)
    expected = log(w1 * exp(logpdf(uniform, x, a, b)) + w2 * exp(logpdf(Gen.Beta(), x, alpha, beta)))
    @test isapprox(actual, expected)

    # test logpdf_grad against finite differencing
    args = (x, [w1, w2], a, b, alpha, beta)
    (x_grad, weights_grad, a_grad, b_grad, alpha_grad, beta_grad) = logpdf_grad(
        uniform_beta_mixture, args...)
    f = (x, weights, a, b, alpha, beta) -> logpdf(uniform_beta_mixture, x, weights, a, b, alpha, beta)

    @test isapprox(x_grad, finite_diff(f, args, 1, dx))
    @test isapprox(weights_grad[1], finite_diff_vec(f, args, 2, 1, dx))
    @test isapprox(weights_grad[2], finite_diff_vec(f, args, 2, 2, dx))
    @test isapprox(a_grad, finite_diff(f, args, 3, dx))
    @test isapprox(b_grad, finite_diff(f, args, 4, dx))
    @test isapprox(alpha_grad, finite_diff(f, args, 5, dx))
    @test isapprox(beta_grad, finite_diff(f, args, 6, dx))
end

mixture_of_normals = @hom_mixture Float64 normal (0, 0)
mixture_of_binomials = @hom_mixture Int binom (0, 0)
mixture_of_mvnormals = @hom_mixture Vector{Float64} mvnormal (1, 2)

@testset "mixture of normals" begin

    @test !is_discrete(mixture_of_normals)
    @test has_output_grad(mixture_of_normals)
    @test has_argument_grads(mixture_of_normals) == (true, true, true)

    w1 = 0.4
    w2 = 0.6
    mu1 = 0.0
    mu2 = 1.0
    std1 = 2.3
    std2 = 1.0

    # random
    x = mixture_of_normals([w1, w2], [mu1, mu2], [std1, std2])

    # logpdf
    x = 1.123
    actual = logpdf(mixture_of_normals, x, [w1, w2], [mu1, mu2], [std1, std2])
    expected = log(w1 * exp(logpdf(normal, x, mu1, std1)) + w2 * exp(logpdf(normal, x, mu2, std2)))
    @test isapprox(actual, expected)

    # test logpdf_grad against finite differencing
    args = (x, [w1, w2], [mu1, mu2], [std1, std2])
    (x_grad, weights_grad, mus_grad, stds_grad) = logpdf_grad(
        mixture_of_normals, args...)
    @test size(mus_grad) == (2,)
    @test size(stds_grad) == (2,)
    f = (x, weights, mus, stds) -> logpdf(mixture_of_normals, x, weights, mus, stds)
    @test isapprox(x_grad, finite_diff(f, args, 1, dx))
    @test isapprox(weights_grad[1], finite_diff_vec(f, args, 2, 1, dx))
    @test isapprox(weights_grad[2], finite_diff_vec(f, args, 2, 2, dx))
    @test isapprox(mus_grad[1], finite_diff_vec(f, args, 3, 1, dx))
    @test isapprox(mus_grad[2], finite_diff_vec(f, args, 3, 2, dx))
    @test isapprox(stds_grad[1], finite_diff_vec(f, args, 4, 1, dx))
    @test isapprox(stds_grad[2], finite_diff_vec(f, args, 4, 2, dx))

    # test that logpdf can be differentiated by ReverseDiff.jl
    # and test against finite differencing
    tp = Gen.new_tape()
    x_tracked = Gen.track(x, tp)
    w1_tracked = Gen.track(w1, tp) 
    w2_tracked = Gen.track(w2, tp) 
    mu1_tracked = Gen.track(mu1, tp) 
    mu2_tracked = Gen.track(mu2, tp) 
    std1_tracked = Gen.track(std1, tp) 
    std2_tracked = Gen.track(std2, tp) 
    lpdf_tracked = logpdf(mixture_of_normals,
        x_tracked, [w1_tracked, w2_tracked], [mu1_tracked, mu2_tracked], [std1_tracked, std2_tracked])
    Gen.deriv!(lpdf_tracked, 1.0)
    Gen.reverse_pass!(tp)
    @test isapprox(Gen.deriv(x_tracked), finite_diff(f, args, 1, dx))
    @test isapprox(Gen.deriv(w1_tracked), finite_diff_vec(f, args, 2, 1, dx))
    @test isapprox(Gen.deriv(w2_tracked), finite_diff_vec(f, args, 2, 2, dx))
    @test isapprox(Gen.deriv(mu1_tracked), finite_diff_vec(f, args, 3, 1, dx))
    @test isapprox(Gen.deriv(mu2_tracked), finite_diff_vec(f, args, 3, 2, dx))
    @test isapprox(Gen.deriv(std1_tracked), finite_diff_vec(f, args, 4, 1, dx))
    @test isapprox(Gen.deriv(std2_tracked), finite_diff_vec(f, args, 4, 2, dx))
end

@testset "mixture of binomial" begin

    @test is_discrete(mixture_of_binomials)
    @test !has_output_grad(mixture_of_binomials)
    @test has_argument_grads(mixture_of_binomials) == (true, false, true)

    w1 = 0.4
    w2 = 0.6
    n1 = 10
    n2 = 20
    p1 = 0.2
    p2 = 0.3

    # random
    x = mixture_of_binomials([w1, w2], [n1, n2], [p1, p2])

    # logpdf
    x = 4
    actual = logpdf(mixture_of_binomials, x, [w1, w2], [n1, n2], [p1, p2])
    expected = log(w1 * exp(logpdf(binom, x, n1, p1)) + w2 * exp(logpdf(binom, x, n2, p2)))
    @test isapprox(actual, expected)

    # test logpdf_grad against finite differencing
    args = (x, [w1, w2], [n1, n2], [p1, p2])
    (x_grad, weights_grad, ns_grad, ps_grad) = logpdf_grad(
        mixture_of_binomials, args...)
    @test ns_grad == nothing
    f = (x, weights, ns, ps) -> logpdf(mixture_of_binomials, x, weights, ns, ps)
    @test isapprox(weights_grad[1], finite_diff_vec(f, args, 2, 1, dx))
    @test isapprox(weights_grad[2], finite_diff_vec(f, args, 2, 2, dx))
    @test isapprox(ps_grad[1], finite_diff_vec(f, args, 4, 1, dx))
    @test isapprox(ps_grad[2], finite_diff_vec(f, args, 4, 2, dx))
end

@testset "mixture of multivariate normals" begin

    @test !is_discrete(mixture_of_mvnormals)
    @test has_output_grad(mixture_of_mvnormals)
    @test has_argument_grads(mixture_of_mvnormals) == (true, true, true)

    w1 = 0.4
    w2 = 0.6
    mu1 = [0.0, 0.0]
    mu2 = [1.0, 1.0]
    cov1 = [2.3 0.0; 0.0 2.3]
    cov2 = [1.0 0.0; 0.0 1.0]

    # random
    mus = cat(mu1, mu2, dims=2)
    covs = cat(cov1, cov2, dims=3)
    x = mixture_of_mvnormals([w1, w2], mus, covs)

    # logpdf
    x = [1.123, -2.123]
    actual = logpdf(mixture_of_mvnormals, x, [w1, w2], mus, covs)
    expected = log(w1 * exp(logpdf(mvnormal, x, mu1, cov1)) + w2 * exp(logpdf(mvnormal, x, mu2, cov2)))
    @test isapprox(actual, expected)

    # test logpdf_grad against finite differencing
    args = (x, [w1, w2], mus, covs)
    (x_grad, weights_grad, mus_grad, covs_grad) = logpdf_grad(
        mixture_of_mvnormals, args...)
    @test size(mus_grad) == (2, 2)
    @test size(covs_grad) == (2, 2, 2)
    f = (x, weights, mus, covs) -> logpdf(mixture_of_mvnormals, x, weights, mus, covs)
    @test isapprox(x_grad[1], finite_diff_vec(f, args, 1, 1, dx))
    @test isapprox(x_grad[2], finite_diff_vec(f, args, 1, 2, dx))
    @test isapprox(weights_grad[1], finite_diff_vec(f, args, 2, 1, dx))
    @test isapprox(weights_grad[2], finite_diff_vec(f, args, 2, 2, dx))
    for idx in 1:length(mus_grad)
        @test isapprox(mus_grad[idx], finite_diff_arr(f, args, 3, idx, dx), rtol=1e-2)
    end

    # to test logpdf_grad with respect to covariance matrices using finite differencing
    # we can't directly introduce differences into the matrix itself, because then
    # it leaves the set of positive definite matrices
    # instead, we define the covariance matrix as a rotated version of some covariance matrix
    # and differentiate with respect to the scalar amount of rotation
    rotmat(theta) = [cos(theta) -sin(theta); sin(theta) cos(theta)]

    @gen function f_cov_model((grad)(theta::Float64))
        cov = [2.0 0.0; 0.0 1.0]
        rot = rotmat(theta)
        x ~ mixture_of_mvnormals([w1, w2], mus, cat(cov, rot * cov * rot', dims=3))
    end

    theta = pi/2
    trace1, = generate(f_cov_model, (theta-dx,), choicemap((:x, x)))
    trace2, = generate(f_cov_model, (theta+dx,), choicemap((:x, x)))
    expected = (get_score(trace2) - get_score(trace1)) / (2 * dx)
    trace, = generate(f_cov_model, (theta,), choicemap((:x, x)))
    ((actual,), _, _) = choice_gradients(trace, select())
    @test isapprox(actual, expected)
end
