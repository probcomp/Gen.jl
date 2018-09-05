@testset "injective" begin

    @inj function foo()
        x = @read(:x)
        y = @read(:y)
        z = @read(:z)
        a = x + y
        b = x - y
        c = x * y * z
        @write(a, :a)
        @write(b, :b)
        @write(c, :c)
    end

    x = 10.123
    y = -1.213
    z = 1.095

    input = DynamicChoiceTrie()
    input[:x] = x
    input[:y] = y
    input[:z] = z

    J = [1.0 1.0 (y*z); 1.0 -1.0 (x*z); 0.0 0.0 (x*y)]
    (expected,) = logabsdet(J)

    (output, logdet) = apply(foo, (), input)
    println(logdet)
    @test isapprox(logdet, expected)

end
