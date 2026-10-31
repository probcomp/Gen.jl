@testset "elliptical slice" begin

    mu = [0., 0.]
    cov = [1. 0.; 0. 1.]
    @gen function foo()
        x = @trace(mvnormal(mu, cov), :x)
        y = @trace(mvnormal(x, [0.01 0.; 0. 0.01]), :y)
    end

    # smoke test
    trace, = generate(foo, (), choicemap((:y, [1., 1.])))
    trace = elliptical_slice(trace, :x, mu, cov)

    # with a constant likelihood, the first proposal on the ellipse is accepted
    @gen function prior_only()
        x = @trace(mvnormal(mu, cov), :x)
    end
    trace, = generate(prior_only, (), choicemap((:x, [0., 0.])))
    for seed in 1:16
        # replay the random choices made by elliptical_slice
        Random.seed!(seed)
        nu = mvnormal(zeros(length(mu)), cov)
        uniform(0, 1)
        theta = uniform(0, 2*pi)
        expected = (trace[:x] .- mu) * cos(theta) + nu * sin(theta) .+ mu
        Random.seed!(seed)
        new_trace = elliptical_slice(trace, :x, mu, cov)
        @test new_trace[:x] ≈ expected
    end

    # check the stationary distribution on a conjugate model:
    # x ~ N(0, 1), y ~ N(x, 1), y = 1, so x | y ~ N(1/2, 1/2)
    Random.seed!(1)
    @gen function bar()
        x = @trace(mvnormal([0.], fill(1., 1, 1)), :x)
        y = @trace(normal(x[1], 1.), :y)
    end
    trace, = generate(bar, (), choicemap((:y, 1.)))
    xs = Float64[]
    for iter=1:5000
        trace = elliptical_slice(trace, :x, [0.], fill(1., 1, 1))
        push!(xs, trace[:x][1])
    end
    x_mean = sum(xs) / length(xs)
    x_var = sum((xs .- x_mean).^2) / length(xs)
    @test isapprox(x_mean, 0.5, atol=0.05)
    @test isapprox(x_var, 0.5, atol=0.05)
end
