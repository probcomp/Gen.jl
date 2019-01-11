@testset "hmc" begin

    # smoke test a function without retval gradient
    @gen function foo()
        x = @addr(normal(0, 1), :x)
        return x
    end

    (trace, _) = initialize(foo, ())
    (new_trace, accepted) = hmc(trace, select(:x))

    # smoke test a function with retval gradient
    @gen (grad) function foo()
        x = @addr(normal(0, 1), :x)
        return x
    end

    (trace, _) = initialize(foo, ())
    (new_trace, accepted) = hmc(trace, select(:x))
end
