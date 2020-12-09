
mixture_of_normals = @mixturedist normal (0, 0) Float64
mixture_of_binomials = @mixturedist binom (0, 0) Int

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
