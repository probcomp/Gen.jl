@testset begin "dynamic selection"

    s = select(:x, :y => :z, :y => :w)

    # test Base.in
    @test :x in s
    @test (:x => :u) in s
    @test (:x => :u => :v) in s
    @test (:y => :z) in s
    @test (:y => :w) in s
    @test !(:y in s)
    @test !((:y => :r) in s)

    # test Base.getindex
    @test s[:x] == AllSelection()
    sub = s[:y]
    @test isa(sub, DynamicSelection)
    @test :z in sub
    @test :w in sub
    @test s[:u] == EmptySelection()
end
