import DataStructures: OrderedDict

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

@testset "scalar normal" begin

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

@testset "zero-dimensional array normal" begin

    # random
    x = normal(0, 1)

    # logpdf_grad
    f = (x, mu, std) -> logpdf(normal, x, mu, std)
    args = (fill(0.4), fill(0.2), fill(0.3))
    actual = logpdf_grad(normal, args...)
    @test isapprox(actual[1], finite_diff(f, args, 1, dx; broadcast=true))
    @test isapprox(actual[2], finite_diff(f, args, 2, dx; broadcast=true))
    @test isapprox(actual[3], finite_diff(f, args, 3, dx; broadcast=true))
end

@testset "array normal" begin

    # random
    x = normal(0, 1)

    # logpdf_grad
    f = (x, mu, std) -> logpdf(normal, x, mu, std)
    args = ([ 0.1   0.2   0.3  ;
              0.4   0.5   0.6  ],
            [ 0.01  0.02  0.03 ;
              0.04  0.05  0.06 ],
            [ 1.    2.    3.   ;
              4.    5.    6.   ])
    actual = logpdf_grad(normal, args...)
    @test isapprox(actual[1], finite_diff(f, args, 1, dx; broadcast=true))
    @test isapprox(actual[2], finite_diff(f, args, 2, dx; broadcast=true))
    @test isapprox(actual[3], finite_diff(f, args, 3, dx; broadcast=true))
end

@testset "array normal with broadcasting" begin

    ## Return shape of `normal`
    @test size(normal([0. 0. 0.], 1.)) == (1, 3)
    @test size(normal(zeros(1, 3, 4), ones(2, 1, 4))) == (2, 3, 4)
    @test_throws DimensionMismatch normal([0 0 0], [1 1])

    ## Return shape of `logpdf` and `logpdf_grad`
    @test size(logpdf(normal,
                      ones(1, 3, 1), ones(2, 1, 1), ones(1, 1, 4))) == (2, 3, 4)
    @test all(size(g) == (2, 3, 4)
              for g in logpdf_grad(
                  normal, ones(1, 3, 1), ones(2, 1, 1), ones(1, 1, 4)))
    # `x` and `mu` are broadcast-incompatible
    @test_throws DimensionMismatch logpdf(normal,
                                          ones(1, 2), ones(1,3), ones(2,1))
    @test_throws DimensionMismatch logpdf_grad(normal,
                                               ones(1, 2), ones(1,3), ones(2,1))
    # `x` and `std` are broadcast-incompatible
    @test_throws DimensionMismatch logpdf(normal,
                                          ones(1, 2), ones(2,1), ones(1,3))
    @test_throws DimensionMismatch logpdf_grad(normal,
                                               ones(1, 2), ones(2,1), ones(1,3))
    # `mu` and `std` are broadcast-incompatible
    @test_throws DimensionMismatch logpdf(normal,
                                          ones(2, 1), ones(1,2), ones(1,3))
    @test_throws DimensionMismatch logpdf_grad(normal,
                                               ones(2, 1), ones(1,2), ones(1,3))

    ## Equivalence of broadcast to operating on bigger arrays
    compact = OrderedDict(:x => fill(0.2, (1, 3, 1)),
                          :mu => fill(0.7, (1, 1, 4)),
                          :std => fill(0.1, (2, 1, 1)))
    expanded = OrderedDict(:x => fill(0.2, (2, 3, 4)),
                           :mu => fill(0.7, (2, 3, 4)),
                           :std => fill(0.1, (2, 3, 4)))
    @test (logpdf(normal, values(compact)...) ==
           logpdf(normal, values(expanded)...))
    @test (logpdf_grad(normal, values(compact)...) ==
           logpdf_grad(normal, values(expanded)...))
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
