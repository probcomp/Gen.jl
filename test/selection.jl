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

@testset begin "addrs"
    choices = choicemap((:a, 1), (:b => :c, 2))
    a = addrs(choices)
    @test a isa Selection
    @test :a in a
    @test (:b => :c) in a
    @test !(:d in a)
    @test get_subselection(a, :b) == select(:c)
    @test length(collect(get_subtrees_shallow(a))) == 2
end

@testset begin "push, merge, merge!"
    x = select(:x, :y => :z)
    push!(x, :y => :w)
    @test !(:y in x)
    @test (:y => :z) in x
    @test (:y => :w) in x
    @test :x in x
    
    y = select(:a => 1 => :b, 5.1, :y => :k)
    z = merge(x, y)
    @test 5.1 in z
    @test :x in z
    @test (:y => :w) in z
    @test (:a => :1 => :b) in z
    @test (:y => :k) in z

    merge!(y, x)
    @test (:y => :w) in y
    @test 5.1 in y
    @test (:y => :k) in y
end

@testset begin "inverted selection"
    sel = select(:x, :y => :z, :a => :b => 1, :a => :b => 2, :a => :c => 1)
    i = invert(sel)
    @test !(:x in i)
    @test !(:y in i)
    @test !((:y => :z) in i)
    @test (:y => :a) in i
    @test !((:a => :b) in i)
    @test (:a => :b => 3) in i
    @test (:a => :d) in i
    @test :z in i

    c = choicemap(
        (:x, 1),
        (:y, 5),
        (:a => :b => 1, 10),
        (:a => :b => 3, 2),
        (:a => :c => 2, 12)
    )
    @test get_selected(c, i) == choicemap((:a => :b => 3, 2), (:a => :c => 2, 12))
end