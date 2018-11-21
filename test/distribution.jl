function finite_diff(f::Function, args::Tuple, i::Int, dx::Float64)
    pos_args = Any[args...]
    pos_args[i] += dx
    neg_args = Any[args...]
    neg_args[i] -= dx
    return (f(pos_args...) - f(neg_args...)) / (2. * dx)
end

const dx = 1e-6

@testset "bernoulli" begin
    f = (x::Bool, prob::Float64) -> logpdf(Gen.Bernoulli(), x, prob)
    args = (false, 0.3,)
    actual = logpdf_grad(Gen.Bernoulli(), args...)
    @test isapprox(actual[2], finite_diff(f, args, 2, dx))
    args = (true, 0.3,)
    actual = logpdf_grad(Gen.Bernoulli(), args...)
    @test isapprox(actual[2], finite_diff(f, args, 2, dx))
end

@testset "beta" begin
    f = (x, alpha, beta) -> logpdf(Gen.Beta(), x, alpha, beta)
    args = (0.4, 0.2, 0.3)
    actual = logpdf_grad(Gen.Beta(), args...)
    @test isapprox(actual[1], finite_diff(f, args, 1, dx))
    @test isapprox(actual[2], finite_diff(f, args, 2, dx))
    @test isapprox(actual[3], finite_diff(f, args, 3, dx))
end

@testset "geometric" begin
    f = (x, prob) -> logpdf(Gen.Geometric(), x, prob)
    args = (3, 0.4)
    actual = logpdf_grad(Gen.Geometric(), args...)
    @test isapprox(actual[2], finite_diff(f, args, 2, dx))
end
