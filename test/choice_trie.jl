@testset "static choice trie to/from array" begin
    inner_leaf_nodes = NamedTuple{(:a, :b), Tuple{Float64,Float64}}((1.0, 2.0))
    inner_internal_nodes = NamedTuple{(), Tuple{}}(())
    inner = StaticChoiceTrie(inner_leaf_nodes, inner_internal_nodes)
    outer_leaf_nodes = NamedTuple{(:c,), Tuple{Float64}}((3.0,))
    outer_internal_nodes = NamedTuple{(:d, :e), Tuple{typeof(inner), typeof(inner)}}((inner, inner))
    outer = StaticChoiceTrie(outer_leaf_nodes, outer_internal_nodes)
    
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

@testset "dynamic choice trie to/from array" begin
    outer = DynamicChoiceTrie()
    set_leaf_node!(outer, :c, 3.)
    inner = DynamicChoiceTrie()
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

@testset "internal node vector choice trie to/from array" begin
    inner = DynamicChoiceTrie()
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
