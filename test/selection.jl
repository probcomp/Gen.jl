@testset "dynamic selection" begin

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
    @test s[:y => :z] == AllSelection()

    # test set_subselection!
    set_subselection!(s, :y, select(:z))
    @test (:y => :z) in s
    @test !((:y => :w) in s)

    selection = select(:x)
    @test :x in selection
    subselection = select(:y)
    set_subselection!(selection, :x, subselection)
    @test (:x => :y) in selection
    @test !(:x in selection)
end

@testset "all selection" begin

    s = selectall()

    # test Base.in
    @test :x in s
    @test (:x => :u) in s

    # test Base.getindex
    @test s[:x] == AllSelection()
    @test s[:x => :y] == AllSelection()
end

@testset "complement selection" begin

    @test !(:x in complement(selectall()))
    @test :x in complement(select())

    @test !(:x in complement(select(:x)))
    @test :y in complement(select(:x))

    @test :x in complement(select(:x => :y => :z))
    @test (:x => :y) in complement(select(:x => :y => :z))
    @test !((:x => :y => :z) in complement(select(:x => :y => :z)))

    @test !(:x in complement(complement(select(:x => :y => :z))))
    @test !((:x => :y) in complement(complement(select(:x => :y => :z))))
    @test (:x => :y => :z) in complement(complement(select(:x => :y => :z)))

    s = complement(select(:x => :y => :z))[:x]
    @test !((:y => :z) in s)
    @test :w in s
    @test :y in s
end
