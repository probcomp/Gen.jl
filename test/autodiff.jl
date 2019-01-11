@testset "autodiff" begin

    @testset "fill" begin
        tp = Gen.new_tape()
        x = Gen.track(2.3, tp)
        y = fill(x, 3)
        grad = rand(3)
        Gen.deriv!(y, grad)
        Gen.reverse_pass!(tp)
        @test isapprox(Gen.deriv(x), sum(grad))

        tp = Gen.new_tape()
        x = Gen.track(2.3, tp)
        y = fill(x, 3, 2)
        grad = rand(3, 2)
        Gen.deriv!(y, grad)
        Gen.reverse_pass!(tp)
        @test isapprox(Gen.deriv(x), sum(grad))
    end

end
