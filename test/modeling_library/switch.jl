@testset "switch combinator" begin

    @gen (grad) function foo((grad)(x::Float64), (grad)(y::Float64))
        @param std::Float64
        z = @trace(normal(x + y, std), :z)
        return z
    end

    @gen (grad) function baz((grad)(x::Float64), (grad)(y::Float64))
        @param std::Float64
        z = @trace(normal(x + 2 * y, std), :z)
        return z
    end

    set_param!(foo, :std, 1.)
    set_param!(baz, :std, 1.)

    bar = Switch(foo, baz)
    args = (1.0, 3.0)

    @testset "simulate" begin
    end

    @testset "generate" begin
    end

    @testset "propose" begin
    end

    @testset "assess" begin
    end

    @testset "update" begin
    end

    @testset "regenerate" begin
    end

    @testset "choice_gradients" begin
    end

    @testset "accumulate_param_gradients!" begin
    end
end
