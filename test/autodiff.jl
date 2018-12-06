import ReverseDiff

@testset "autodiff" begin

    @testset "fill" begin
        tp = ReverseDiff.InstructionTape()
        x = ReverseDiff.track(2.3, tp)
        y = fill(x, 3)
        grad = rand(3)
        ReverseDiff.deriv!(y, grad)
        ReverseDiff.reverse_pass!(tp)
        @test isapprox(ReverseDiff.deriv(x), sum(grad))

        tp = ReverseDiff.InstructionTape()
        x = ReverseDiff.track(2.3, tp)
        y = fill(x, 3, 2)
        grad = rand(3, 2)
        ReverseDiff.deriv!(y, grad)
        ReverseDiff.reverse_pass!(tp)
        @test isapprox(ReverseDiff.deriv(x), sum(grad))
    end

end
