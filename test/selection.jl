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

@testset begin "all selection"

    s = selectall()

    # test Base.in
    @test :x in s
    @test (:x => :u) in s

    # test Base.getindex
    @test s[:x] == AllSelection()
    @test s[:x => :y] == AllSelection()
end

@testset begin "complement selection"

    @test !(:x in ComplementSelection(selectall()))
    @test :x in ComplementSelection(select())

    @test !(:x in ComplementSelection(select(:x)))
    @test :y in ComplementSelection(select(:x))

    @test :x in ComplementSelection(select(:x => :y => :z))
    @test (:x => :y) in ComplementSelection(select(:x => :y => :z))
    @test !((:x => :y => :z) in ComplementSelection(select(:x => :y => :z)))

    @test !(:x in ComplementSelection(ComplementSelection(select(:x => :y => :z))))
    @test !((:x => :y) in ComplementSelection(ComplementSelection(select(:x => :y => :z))))
    @test (:x => :y => :z) in ComplementSelection(ComplementSelection(select(:x => :y => :z)))

    s = ComplementSelection(select(:x => :y => :z))[:x]
    @test !((:y => :z) in s)
    @test :w in s
    @test :y in s
end
