discrete_product = ProductDistribution(bernoulli, binom)

@testset "product of discrete distributions" begin
    @test is_discrete(discrete_product)
    grad_bools = (has_output_grad(discrete_product), has_argument_grads(discrete_product)...)
    @test grad_bools == (false, true, false, true)

    p1 = 0.5
    (n, p2) = (3, 0.9)

    # random
    x = discrete_product(p1, n, p2)
    @assert typeof(x) == Gen.get_return_type(discrete_product) == Tuple{Bool, Int}

    # logpdf
    x = (true, 2)
    actual = logpdf(discrete_product, x, p1, n, p2)
    expected = logpdf(bernoulli, x[1], p1) + logpdf(binom, x[2], n, p2)
    @test isapprox(actual, expected)

    # test logpdf_grad against finite differencing
    f = (x, p1, n, p2) -> logpdf(discrete_product, x, p1, n, p2)
    args = (x, p1, n, p2)
    actual = logpdf_grad(discrete_product, args...)
    for (i, b) in enumerate(grad_bools)
        if b
            @test isapprox(actual[i], finite_diff(f, args, i, dx))
        end
    end
end

continuous_product = ProductDistribution(uniform, normal)

@testset "product of continuous distributions" begin
    @test !is_discrete(continuous_product)
    grad_bools = (has_output_grad(continuous_product), has_argument_grads(continuous_product)...)
    @test grad_bools == (true, true, true, true, true)

    (low, high) = (-0.5, 0.5)
    (mu, std) = (0.0, 1.0)

    # random
    x = continuous_product(low, high, mu, std)
    @assert typeof(x) == Gen.get_return_type(continuous_product) == Typle{Float64, Float64}

    # logpdf
    x = (0.1, 0.7)
    actual = logpdf(continuous_product, x, low, high, mu, std)
    expected = logpdf(uniform, x[1], low, high) + logpdf(normal, x[2], mu, std)
    @test isapprox(actual, expected)

    # test logpdf_grad against finite differencing
    f = (x, low, high, mu, std) -> logpdf(continuous_product, x, low, high, mu, std)
    args = (x, low, high, mu, std)
    actual = logpdf_grad(continuous_product, args...)
    for (i, b) in enumerate(grad_bools)
        if b
            @test isapprox(actual[i], finite_diff(f, args, i, dx))
        end
    end
end

dissimilar_product = ProductDistribution(bernoulli, normal)

@testset "product of dissimilarly-typed distributions" begin
    @test !is_discrete(dissimilar_product)
    grad_bools = (has_output_grad(dissimilar_product), has_argument_grads(dissimilar_product)...)
    @test grad_bools == (false, true, true, true)

    p = 0.5
    (mu, std) = (0.0, 1.0)

    # random
    x = dissimilar_product(p, mu, std)
    @assert typeof(x) == Gen.get_return_type(dissimilar_product) == Tuple{Bool, Float64}

    # logpdf
    x = (false, 0.3)
    actual = logpdf(dissimilar_product, x, p, mu, std)
    expected = logpdf(bernoulli, x[1], p) + logpdf(normal, x[2], mu, std)
    @test isapprox(actual, expected)

    # test logpdf_grad against finite differencing
    f = (x, p, mu, std) -> logpdf(dissimilar_product, x, p, mu, std)
    args = (x, p, mu, std)
    actual = logpdf_grad(dissimilar_product, args...)
    for (i, b) in enumerate(grad_bools)
        if b
            @test isapprox(actual[i], finite_diff(f, args, i, dx))
        end
    end
end
