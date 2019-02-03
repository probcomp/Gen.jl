@testset "mala" begin

    # smoke test a function without retval gradient
    @gen function foo()
        x = @addr(normal(0, 1), :x)
        return x
    end

    (trace, _) = generate(foo, ())
    (new_trace, accepted) = mala(trace, select(:x), 0.1)

    # smoke test a function with retval gradient
    @gen (grad) function foo()
        x = @addr(normal(0, 1), :x)
        return x
    end

    (trace, _) = generate(foo, ())
    (new_trace, accepted) = mala(trace, select(:x), 0.1)
end
