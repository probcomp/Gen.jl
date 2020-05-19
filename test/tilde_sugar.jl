using Gen
import MacroTools

normalize(ex) = MacroTools.prewalk(MacroTools.rmlines, ex)



# dynamic
@testset "tilde syntax smoke test (dynamic)" begin
    @gen function foo(a, b)
        p ~ beta(a, b)
        return ({:coin => 1} ~ bernoulli(p)) + ({:coin => 2} ~ bernoulli(p))
    end

    trace = simulate(foo, (1, 1))
    @test (trace[:coin => 1] + trace[:coin => 2]) == get_retval(trace)
    @test get_score(trace) == sum([(trace[:coin => i] ? log(trace[:p]) : log(1-trace[:p])) for i=1:2])
end

@testset "tilde syntax desugars as expected (dynamic)" begin
    expected = normalize(:(
    @gen function foo()
        x = @trace(normal(0, 1), :x)
        y = @trace(normal(0, 1), :y => :z)
        z = @trace(bar())
    end))

    actual = normalize(Gen.desugar_tildes(:(
    @gen function foo()
        x ~ normal(0, 1)
        y = ({:y => :z} ~ normal(0, 1))
        z = ({*} ~ bar())
    end)))

    @test actual == expected
end

# static


@testset "tilde syntax smoke test (static)" begin
    @gen (static) function bar(r)
        a ~ normal(0, 1)
        return r
    end

    @gen (static) function foo()
        x ~ bar(1)
        yz = {:y => :z} ~ bar(2)
        u ~ normal(0, 1)
        {:v => :w} ~ normal(0, 1)
        retrate = 7
        return {:ret} ~ poisson(retrate)
    end

    Gen.load_generated_functions()

    trace = simulate(foo, ())

    # random choices
    @test trace[:u] isa Float64
    @test trace[:v => :w] isa Float64
    @test trace[:x => :a] isa Float64
    @test trace[:y => :z => :a] isa Float64

    # auxiliary state
    @test trace[:x] == 1
    @test trace[:y => :z] == 2

    # return value
    @test trace[] == trace[:ret]
end


@testset "tilde syntax desugars as expected (static)" begin
expected = normalize(:(
@gen (static) function foo()
    x = @trace(normal(0, 1), :x)
    y = @trace(normal(0, 1), :y)
end))

actual = normalize(Gen.desugar_tildes(:(
@gen (static) function foo()
    x ~ normal(0, 1)
    y = ({:y} ~ normal(0, 1))
end)))

@test actual == expected
end
