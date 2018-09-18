function finite_diff(f::Function, args::Tuple, i::Int, dx::Float64)
    pos_args = [args...]
    pos_args[i] += dx
    neg_args = [args...]
    neg_args[i] -= dx
    return (f(pos_args...) - f(neg_args...)) / (2. * dx)
end

const dx = 1e-6

@testset "beta" begin
    f = (x, alpha, beta) -> logpdf(Gen.Beta(), x, alpha, beta)
    args = (0.4, 0.2, 0.3)
    actual = logpdf_grad(Gen.Beta(), args...)
    @test isapprox(actual[1], finite_diff(f, args, 1, dx))
    @test isapprox(actual[2], finite_diff(f, args, 2, dx))
    @test isapprox(actual[3], finite_diff(f, args, 3, dx))
end
