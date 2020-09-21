# y ~ @gaussian_noise_around x
macro gaussian_noise_around(x)
    :(normal($(esc(x)), 1.0))
end

# y = @traced_gaussian_noise_around(x, :y)
macro traced_gaussian_noise_around(x, y)
    :(
        Gen.@trace(
            normal( $(esc(x)) , 1.0),
            $y
        )
    )
end

# @add_gaussian_noise x y  # means y ~ normal(x, 1.)
macro add_gaussian_noise(x, y)
    Expr(:call, esc(:~), esc(y), :(normal($(esc(x)), 1.0)))
    #:($(esc(y)) $(esc(:~)) normal($(esc(x)), 1.0))
end

macro set_x_to_normal_tilde(mean)
    Expr(:call, esc(:~), esc(:x), :(normal($(esc(mean)), 1)))
    #:($(esc(:x)) $(esc(:~)) normal($(esc(mean)), 1))
end

macro set_x_to_normal_trace(mean)
    :($(esc(:x)) = Gen.@trace(normal($(esc(mean)), 1), $(QuoteNode(:x))))
end

macro set_x_to_normal_traceexpr(mean)
    :($(esc(:x)) = $(Expr(:gentrace,
        :(normal($(esc(mean)), 1.)),
        QuoteNode(:x)
    )))
end

@testset "macros in dynamic DSL" begin
    @gen function foo1()
        x ~ exponential(1)
        y ~ @gaussian_noise_around x # y ~ normal(x, 1)
    end
    trace = simulate(foo1, ())
    @test trace[:y] != trace[:x]

    @gen function foo2()
        x ~ exponential(1)
        y = Gen.@trace(@gaussian_noise_around(x), :y) # y = @trace(normal(x, 1), :y)
    end
    trace = simulate(foo2, ())
    @test trace[:y] != trace[:x]

    @gen function foo3()
        x ~ exponential(1)
        y = @traced_gaussian_noise_around x :y # y = @trace(normal(x, 1), :y)
    end
    trace = simulate(foo3, ())
    @test trace[:y] != trace[:x]

    @gen function foo4()
        x ~ exponential(1)
        @add_gaussian_noise x y # y ~ normal(x, 1)
    end
    trace = simulate(foo4, ())
    @test trace[:y] != trace[:x]

    @gen function foo5()
        y ~ exponential(1)
        @set_x_to_normal_tilde y # x ~ normal(y, 1)
    end
    trace = simulate(foo5, ())
    @test trace[:y] != trace[:x]

    @gen function foo6()
        y ~ exponential(1)
        @set_x_to_normal_trace y # x = Gen.@trace(normal(y, 1), :y)
    end
    trace = simulate(foo6, ())
    @test trace[:y] != trace[:x]

    @gen function foo7()
        y ~ exponential(1)
        @set_x_to_normal_traceexpr y # x = Gen.@trace(normal(y, 1), :y) USING EXPR(:GENTRACE)
    end
    trace = simulate(foo7, ())
    @test trace[:y] != trace[:x]
end

############################################
# Define static functions at global scope: #
############################################

@gen (static) function foo1()
    x ~ exponential(1)
    y ~ @gaussian_noise_around x # y ~ normal(x, 1)
end
@gen (static) function foo2()
    x ~ exponential(1)
    y = Gen.@trace(@gaussian_noise_around(x), :y) # y = @trace(normal(x, 1), :y)
end
@gen (static) function foo3()
    x ~ exponential(1)
    y = @traced_gaussian_noise_around x :y # y = @trace(normal(x, 1), :y)
end
@gen (static) function foo4()
    x ~ exponential(1)
    @add_gaussian_noise x y # y ~ normal(x, 1)
end
@gen (static) function foo5()
    y ~ exponential(1)
    @set_x_to_normal_tilde y # x ~ normal(y, 1)
end
@gen (static) function foo6()
    y ~ exponential(1)
    @set_x_to_normal_trace y # x = Gen.@trace(normal(y, 1), :y)
end
@gen (static) function foo7()
    y ~ exponential(1)
    @set_x_to_normal_traceexpr y # x = Gen.@trace(normal(y, 1), :y) USING EXPR(:GENTRACE)
end

@load_generated_functions()

@testset "macros in static DSL" begin
    trace = simulate(foo1, ())
    @test trace[:y] != trace[:x]

    trace = simulate(foo2, ())
    @test trace[:y] != trace[:x]

    trace = simulate(foo3, ())
    @test trace[:y] != trace[:x]

    trace = simulate(foo4, ())
    @test trace[:y] != trace[:x]

    trace = simulate(foo5, ())
    @test trace[:y] != trace[:x]

    trace = simulate(foo6, ())
    @test trace[:y] != trace[:x]

    trace = simulate(foo7, ())
    @test trace[:y] != trace[:x]
end