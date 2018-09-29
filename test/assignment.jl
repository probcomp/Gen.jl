@testset "static assignment to/from array" begin
    inner = StaticAssignment((a=1., b=2.),NamedTuple())
    outer = StaticAssignment((c=3.,), (d=inner, e=inner))
    
    arr = to_array(outer, Float64)
    @test to_array(outer, Float64) == Float64[3.0, 1.0, 2.0, 1.0, 2.0]
    
    trie = from_array(outer, Float64[1, 2, 3, 4, 5])
    @test trie[:c] == 1.0
    @test trie[:d => :a] == 2.0
    @test trie[:d => :b] == 3.0
    @test trie[:e => :a] == 4.0
    @test trie[:e => :b] == 5.0
    @test length(get_internal_nodes(trie)) == 2
    @test length(get_leaf_nodes(trie)) == 1
    inner1 = get_internal_node(trie, :d)
    @test length(get_leaf_nodes(inner1)) == 2
    @test length(get_internal_nodes(inner1)) == 0
    inner2 = get_internal_node(trie, :e)
    @test length(get_leaf_nodes(inner2)) == 2
    @test length(get_internal_nodes(inner2)) == 0
end

@testset "dynamic assignment to/from array" begin
    outer = DynamicAssignment()
    set_leaf_node!(outer, :c, 3.)
    inner = DynamicAssignment()
    set_leaf_node!(inner, :a, 1.)
    set_leaf_node!(inner, :b, 2.)
    set_internal_node!(outer, :d, inner)
    set_internal_node!(outer, :e, inner)
    
    arr = to_array(outer, Float64)
    @test to_array(outer, Float64) == Float64[3.0, 1.0, 2.0, 1.0, 2.0]
    
    trie = from_array(outer, Float64[1, 2, 3, 4, 5])
    @test trie[:c] == 1.0
    @test trie[:d => :a] == 2.0
    @test trie[:d => :b] == 3.0
    @test trie[:e => :a] == 4.0
    @test trie[:e => :b] == 5.0
    @test length(get_internal_nodes(trie)) == 2
    @test length(get_leaf_nodes(trie)) == 1
    inner1 = get_internal_node(trie, :d)
    @test length(get_leaf_nodes(inner1)) == 2
    @test length(get_internal_nodes(inner1)) == 0
    inner2 = get_internal_node(trie, :e)
    @test length(get_leaf_nodes(inner2)) == 2
    @test length(get_internal_nodes(inner2)) == 0
end

@testset "internal node vector assignment to/from array" begin
    inner = DynamicAssignment()
    set_leaf_node!(inner, :a, 1.)
    set_leaf_node!(inner, :b, 2.)
    outer = vectorize_internal([inner, inner, inner])

    arr = to_array(outer, Float64)
    @test to_array(outer, Float64) == Float64[1, 2, 1, 2, 1, 2]

    trie = from_array(outer, Float64[1, 2, 3, 4, 5, 6])
    @test trie[1 => :a] == 1.0
    @test trie[1 => :b] == 2.0
    @test trie[2 => :a] == 3.0
    @test trie[2 => :b] == 4.0
    @test trie[3 => :a] == 5.0
    @test trie[3 => :b] == 6.0
    @test length(get_internal_nodes(trie)) == 3
end

@testset "dynamic assignment merge" begin
    inner = DynamicAssignment()
    set_leaf_node!(inner, :x, 1)
    trie1 = DynamicAssignment()
    set_leaf_node!(trie1, :a, 1.)
    set_leaf_node!(trie1, :b, 2.)
    set_internal_node!(trie1, :c, inner)
    set_internal_node!(trie1, :shared, inner)
    trie2 = DynamicAssignment()
    set_leaf_node!(trie2, :d, 3.)
    set_internal_node!(trie2, :e, inner)
    set_internal_node!(trie2, :f, inner)
    inner2 = DynamicAssignment()
    set_leaf_node!(inner2, :y, 4.)
    set_internal_node!(trie2, :shared, inner2)
    trie = merge(trie1, trie2)
    @test trie[:a] == 1.
    @test trie[:b] == 2.
    @test trie[:d] == 3.
    @test trie[:c => :x] == 1
    @test trie[:e => :x] == 1
    @test trie[:f => :x] == 1
    @test trie[:shared => :x] == 1
    @test trie[:shared => :y] == 4.
    @test length(get_internal_nodes(trie)) == 4
    @test length(get_leaf_nodes(trie)) == 3
end

@testset "static assignment merge" begin
    inner = DynamicAssignment()
    set_leaf_node!(inner, :x, 1)
    inner2 = DynamicAssignment()
    set_leaf_node!(inner2, :y, 4.)
    trie1 = StaticAssignment((a=1., b=2.), (c=inner, shared=inner))
    trie2 = StaticAssignment((d=3.,), (e=inner, f=inner, shared=inner2))
    trie = merge(trie1, trie2)
    @test trie[:a] == 1.
    @test trie[:b] == 2.
    @test trie[:d] == 3.
    @test trie[:c => :x] == 1
    @test trie[:e => :x] == 1
    @test trie[:f => :x] == 1
    @test trie[:shared => :x] == 1
    @test trie[:shared => :y] == 4.
    @test length(get_internal_nodes(trie)) == 4
    @test length(get_leaf_nodes(trie)) == 3
end
