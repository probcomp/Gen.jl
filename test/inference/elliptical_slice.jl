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
end
