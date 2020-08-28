@testset "importance sampling" begin

    @gen function model()
        x = @trace(normal(0, 1), :x)
        @trace(normal(x, 1), :y)
    end

    @gen function proposal()
        @trace(normal(0, 2), :x)
    end

    y = 2.
    observations = choicemap()
    set_value!(observations, :y, y)

    n = 4

    (traces, log_weights, lml_est) = importance_sampling(model, (), observations, n)
    @test length(traces) == n
    @test length(log_weights) == n
    @test isapprox(logsumexp(log_weights), 0., atol=1e-14)
    @test !isnan(lml_est)
    for trace in traces
        @test get_choices(trace)[:y] == y
    end

    (traces, log_weights, lml_est) = importance_sampling(model, (), observations, proposal, (), n)
    @test length(traces) == n
    @test length(log_weights) == n
    @test isapprox(logsumexp(log_weights), 0., atol=1e-14)
    @test !isnan(lml_est)
    for trace in traces
        @test get_choices(trace)[:y] == y
    end

    (trace, lml_est) = importance_resampling(model, (), observations, n)
    @test isapprox(logsumexp(log_weights), 0., atol=1e-14)
    @test !isnan(lml_est)
    @test get_choices(trace)[:y] == y

    (trace, lml_est) = importance_resampling(model, (), observations, proposal, (), n)
    @test isapprox(logsumexp(log_weights), 0., atol=1e-14)
    @test !isnan(lml_est)
    @test get_choices(trace)[:y] == y
end


