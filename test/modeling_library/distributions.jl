@testset "bernoulli" begin
    
    # random
    x = bernoulli(0.5)
    
    # logpdf_grad
    f = (x::Bool, prob::Float64) -> logpdf(bernoulli, x, prob)
    args = (false, 0.3,)
    actual = logpdf_grad(bernoulli, args...)
    @test isapprox(actual[2], finite_diff(f, args, 2, dx))
    args = (true, 0.3,)
    actual = logpdf_grad(bernoulli, args...)
    @test isapprox(actual[2], finite_diff(f, args, 2, dx))
end

@testset "beta" begin

    # random
    x = beta(0.5, 0.5)
    @test 0 < x < 1

    # logpdf_grad
    f = (x, alpha, beta_param) -> logpdf(beta, x, alpha, beta_param)
    args = (0.4, 0.2, 0.3)
    actual = logpdf_grad(beta, args...)
    @test isapprox(actual[1], finite_diff(f, args, 1, dx))
    @test isapprox(actual[2], finite_diff(f, args, 2, dx))
    @test isapprox(actual[3], finite_diff(f, args, 3, dx))
end

@testset "categorical" begin

    # random
    x = categorical([0.2, 0.3, 0.5])
    @test 0 < x < 4

    # logpdf_grad
    f = (x, probs) -> logpdf(categorical, x, probs)
    args = (2, [0.2, 0.3, 0.5])
    actual = logpdf_grad(categorical, args...)
    @test actual[1] == nothing
    @test isapprox(actual[2][1], finite_diff_vec(f, args, 2, 1, dx))
    @test isapprox(actual[2][2], finite_diff_vec(f, args, 2, 2, dx))
    @test isapprox(actual[2][3], finite_diff_vec(f, args, 2, 3, dx))
end


@testset "gamma" begin

    # random
    x = gamma(1, 1)
    @test 0 < x

    # logpdf_grad
    f = (x, shape, scale) -> logpdf(gamma, x, shape, scale)
    args = (0.4, 0.2, 0.3)
    actual = logpdf_grad(gamma, args...)
    @test isapprox(actual[1], finite_diff(f, args, 1, dx))
    @test isapprox(actual[2], finite_diff(f, args, 2, dx))
    @test isapprox(actual[3], finite_diff(f, args, 3, dx))
end

@testset "normal" begin

    # random
    x = normal(0, 1)

    # logpdf_grad
    f = (x, mu, std) -> logpdf(normal, x, mu, std)
    args = (0.4, 0.2, 0.3)
    actual = logpdf_grad(normal, args...)
    @test isapprox(actual[1], finite_diff(f, args, 1, dx))
    @test isapprox(actual[2], finite_diff(f, args, 2, dx))
    @test isapprox(actual[3], finite_diff(f, args, 3, dx))
end

@testset "multivariate normal" begin

    # random
    x = mvnormal([0.0, 0.0], [1.0 0.2; 0.2 1.4])
    @test length(x) == 2

    # logpdf_grad
    f = (x, mu, cov) -> logpdf(mvnormal, x, mu, cov)
    args = ([0.1, 0.2], [0.3, 0.4], [1.0 0.2; 0.2 1.4])
    actual = logpdf_grad(mvnormal, args...)
    @test isapprox(actual[1][1], finite_diff_vec(f, args, 1, 1, dx))
    @test isapprox(actual[1][2], finite_diff_vec(f, args, 1, 2, dx))
    @test actual[2] === nothing # not yet implemented 
    @test actual[3] === nothing # not yet implemented
end

@testset "piecewise_uniform" begin
   
    # random
    x = piecewise_uniform([-0.5, 0.5], [1.0])
    @test -0.5 < x < 0.5

    # logpdf_grad
    f = (x, bounds, probs) -> logpdf(piecewise_uniform, x, bounds, probs)
    args = (0.5, [-1.0, 0.0, 1.0], [0.4, 0.6])
    actual = logpdf_grad(piecewise_uniform, args...)
    @test isapprox(actual[1], finite_diff(f, args, 1, dx))
    @test isapprox(actual[2][1], finite_diff_vec(f, args, 2, 1, dx))
    @test isapprox(actual[2][2], finite_diff_vec(f, args, 2, 2, dx))
    @test isapprox(actual[2][3], finite_diff_vec(f, args, 2, 3, dx))
    @test isapprox(actual[3][1], finite_diff_vec(f, args, 3, 1, dx))
    @test isapprox(actual[3][2], finite_diff_vec(f, args, 3, 2, dx))
end

@testset "beta uniform mixture" begin

    # random
    x = beta_uniform(0.5, 0.5, 0.5)
    @test 0 < x < 1
    
    # logpdf_grad
    f = (x, theta, alpha, beta) -> logpdf(beta_uniform, x, theta, alpha, beta)
    args = (0.5, 0.4, 10., 2.)
    actual = logpdf_grad(beta_uniform, args...)
    @test isapprox(actual[1], finite_diff(f, args, 1, dx))
    @test isapprox(actual[2], finite_diff(f, args, 2, dx))
    @test isapprox(actual[3], finite_diff(f, args, 3, dx))
    @test isapprox(actual[4], finite_diff(f, args, 4, dx))
end

@testset "geometric" begin

    # random
    @test geometric(0.5) >= 0

    # logpdf_grad
    f = (x, p) -> logpdf(geometric, x, p)
    args = (4, 0.3)
    actual = logpdf_grad(geometric, args...)
    @test actual[1] == nothing
    @test isapprox(actual[2], finite_diff(f, args, 2, dx))
end

@testset "exponential" begin

    # random
    @test exponential(0.5) > 0

    # logpdf_grad
    f = (x, rate) -> logpdf(exponential, x, rate)
    args = (1.2, 0.5)
    actual = logpdf_grad(exponential, args...)
    @test isapprox(actual[1], finite_diff(f, args, 1, dx))
    @test isapprox(actual[2], finite_diff(f, args, 2, dx))
end
