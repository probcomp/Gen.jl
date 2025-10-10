@testset "hmc" begin
    import LinearAlgebra
    # smoke test a function without retval gradient
    @gen function foo()
        x = @trace(normal(0, 1), :x)
        return x
    end

    (trace, _) = generate(foo, ())
    (new_trace, accepted) = hmc(trace, select(:x))

    # smoke test a function with retval gradient
    @gen (grad) function foo()
        x = @trace(normal(0, 1), :x)
        return x
    end

    (trace, _) = generate(foo, ())
    (new_trace, accepted) = hmc(trace, select(:x))

    # smoke test with vector metric
    @gen function bar()
        x = @trace(normal(0, 1), :x)
        y = @trace(normal(0, 1), :y)
        return x + y
    end

    (trace, _) = generate(bar, ())
    metric_vec = [1.0, 2.0]
    (new_trace, accepted) = hmc(trace, select(:x, :y); metric=metric_vec)

    # smoke test with Diagonal metric
    (trace, _) = generate(bar, ())
    metric_diag = LinearAlgebra.Diagonal([1.0, 2.0])
    (new_trace, accepted) = hmc(trace, select(:x, :y); metric=metric_diag)

    # smoke test with Dense matrix metric
    (trace, _) = generate(bar, ())
    metric_dense = [1.0 0.1; 0.1 2.0]
    (new_trace, accepted) = hmc(trace, select(:x, :y); metric=metric_dense)

    # smoke test with vector metric and retval gradient
    @gen (grad) function bar_grad()
        x = @trace(normal(0, 1), :x)
        y = @trace(normal(0, 1), :y)
        return x + y
    end

    (trace, _) = generate(bar_grad, ())
    metric_vec = [0.5, 1.5]
    (new_trace, accepted) = hmc(trace, select(:x, :y); metric=metric_vec)

    # smoke test with Diagonal metric and retval gradient
    (trace, _) = generate(bar_grad, ())
    metric_diag = LinearAlgebra.Diagonal([0.5, 1.5])
    (new_trace, accepted) = hmc(trace, select(:x, :y); metric=metric_diag)

    # smoke test with Dense matrix metric and retval gradient
    (trace, _) = generate(bar_grad, ())
    metric_dense = [0.5 0.2; 0.2 1.5]
    (new_trace, accepted) = hmc(trace, select(:x, :y); metric=metric_dense)
end
