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
    @test isa(sub, DynamicAddressTree)
    @test :z in sub
    @test :w in sub
    @test s[:u] == EmptyAddressTree()
    @test s[:y => :z] == AllSelection()

    # test set_subselection!
    set_subtree!(s, :y, select(:z))
    @test (:y => :z) in s
    @test !((:y => :w) in s)

    selection = select(:x)
    @test :x in selection
    subselection = select(:y)
    set_subtree!(selection, :x, subselection)
    @test (:x => :y) in selection
    @test !(:x in selection)
end

@testset begin "all selection"

    s = AllSelection()

    # test Base.in
    @test :x in s
    @test (:x => :u) in s

    # test Base.getindex
    @test s[:x] == AllSelection()
    @test s[:x => :y] == AllSelection()
end