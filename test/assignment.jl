@testset "static assignment to/from array" begin
    subassmt = StaticAssignment((a=1., b=2.),NamedTuple())
    outer = StaticAssignment((c=3.,), (d=subassmt, e=subassmt))
    
    arr = to_array(outer, Float64)
    @test to_array(outer, Float64) == Float64[3.0, 1.0, 2.0, 1.0, 2.0]
    
    assmt = from_array(outer, Float64[1, 2, 3, 4, 5])
    @test assmt[:c] == 1.0
    @test assmt[:d => :a] == 2.0
    @test assmt[:d => :b] == 3.0
    @test assmt[:e => :a] == 4.0
    @test assmt[:e => :b] == 5.0
    @test length(collect(get_subassmts_shallow(assmt))) == 2
    @test length(collect(get_values_shallow(assmt))) == 1
    subassmt1 = get_subassmt(assmt, :d)
    @test length(collect(get_values_shallow(subassmt1))) == 2
    @test length(collect(get_subassmts_shallow(subassmt1))) == 0
    subassmt2 = get_subassmt(assmt, :e)
    @test length(collect(get_values_shallow(subassmt2))) == 2
    @test length(collect(get_subassmts_shallow(subassmt2))) == 0
end

@testset "dynamic assignment to/from array" begin
    outer = DynamicAssignment()
    set_value!(outer, :c, 3.)
    subassmt = DynamicAssignment()
    set_value!(subassmt, :a, 1.)
    set_value!(subassmt, :b, 2.)
    set_subassmt!(outer, :d, subassmt)
    set_subassmt!(outer, :e, subassmt)
    
    arr = to_array(outer, Float64)
    @test to_array(outer, Float64) == Float64[3.0, 1.0, 2.0, 1.0, 2.0]
    
    assmt = from_array(outer, Float64[1, 2, 3, 4, 5])
    @test assmt[:c] == 1.0
    @test assmt[:d => :a] == 2.0
    @test assmt[:d => :b] == 3.0
    @test assmt[:e => :a] == 4.0
    @test assmt[:e => :b] == 5.0
    @test length(collect(get_subassmts_shallow(assmt))) == 2
    @test length(collect(get_values_shallow(assmt))) == 1
    subassmt1 = get_subassmt(assmt, :d)
    @test length(collect(get_values_shallow(subassmt1))) == 2
    @test length(collect(get_subassmts_shallow(subassmt1))) == 0
    subassmt2 = get_subassmt(assmt, :e)
    @test length(collect(get_values_shallow(subassmt2))) == 2
    @test length(collect(get_subassmts_shallow(subassmt2))) == 0
end

@testset "internal vector assignment to/from array" begin
    inner = DynamicAssignment()
    set_value!(inner, :a, 1.)
    set_value!(inner, :b, 2.)
    outer = vectorize_internal([inner, inner, inner])

    arr = to_array(outer, Float64)
    @test to_array(outer, Float64) == Float64[1, 2, 1, 2, 1, 2]

    assmt = from_array(outer, Float64[1, 2, 3, 4, 5, 6])
    @test assmt[1 => :a] == 1.0
    @test assmt[1 => :b] == 2.0
    @test assmt[2 => :a] == 3.0
    @test assmt[2 => :b] == 4.0
    @test assmt[3 => :a] == 5.0
    @test assmt[3 => :b] == 6.0
    @test length(collect(get_subassmts_shallow(assmt))) == 3
end

@testset "dynamic assignment merge" begin
    subassmt = DynamicAssignment()
    set_value!(subassmt, :x, 1)
    assmt1 = DynamicAssignment()
    set_value!(assmt1, :a, 1.)
    set_value!(assmt1, :b, 2.)
    set_subassmt!(assmt1, :c, subassmt)
    set_subassmt!(assmt1, :shared, subassmt)
    assmt2 = DynamicAssignment()
    set_value!(assmt2, :d, 3.)
    set_subassmt!(assmt2, :e, subassmt)
    set_subassmt!(assmt2, :f, subassmt)
    subassmt2 = DynamicAssignment()
    set_value!(subassmt2, :y, 4.)
    set_subassmt!(assmt2, :shared, subassmt2)
    assmt = merge(assmt1, assmt2)
    @test assmt[:a] == 1.
    @test assmt[:b] == 2.
    @test assmt[:d] == 3.
    @test assmt[:c => :x] == 1
    @test assmt[:e => :x] == 1
    @test assmt[:f => :x] == 1
    @test assmt[:shared => :x] == 1
    @test assmt[:shared => :y] == 4.
    @test length(collect(get_subassmts_shallow(assmt))) == 4
    @test length(collect(get_values_shallow(assmt))) == 3
end

@testset "static assignment merge" begin
    subassmt = DynamicAssignment()
    set_value!(subassmt, :x, 1)
    subassmt2 = DynamicAssignment()
    set_value!(subassmt2, :y, 4.)
    assmt1 = StaticAssignment((a=1., b=2.), (c=subassmt, shared=subassmt))
    assmt2 = StaticAssignment((d=3.,), (e=subassmt, f=subassmt, shared=subassmt2))
    assmt = merge(assmt1, assmt2)
    @test assmt[:a] == 1.
    @test assmt[:b] == 2.
    @test assmt[:d] == 3.
    @test assmt[:c => :x] == 1
    @test assmt[:e => :x] == 1
    @test assmt[:f => :x] == 1
    @test assmt[:shared => :x] == 1
    @test assmt[:shared => :y] == 4.
    @test length(collect(get_subassmts_shallow(assmt))) == 4
    @test length(collect(get_values_shallow(assmt))) == 3
end

@testset "static assignment errors" begin

    # get_assmt on an address that contains a value throws a KeyError
    assmt = StaticAssignment((x=1,), NamedTuple())
    threw = false
    try get_subassmt(assmt, :x) catch KeyError threw = true end
    @test threw

    # static_get_subassmt on an address that contains a value throws a KeyError
    assmt = StaticAssignment((x=1,), NamedTuple())
    threw = false
    try static_get_subassmt(assmt, Val(:x)) catch KeyError threw = true end
    @test threw

    # get_assmt on an address whose prefix contains a value throws a KeyError
    assmt = StaticAssignment((x=1,), NamedTuple())
    threw = false
    try get_subassmt(assmt, :x => :y) catch KeyError threw = true end
    @test threw

    # static_get_assmt on an address whose prefix contains a value throws a KeyError
    assmt = StaticAssignment((x=1,), NamedTuple())
    threw = false
    try static_get_subassmt(assmt, Val(:x)) catch KeyError threw = true end
    @test threw

    # get_assmt on an address that contains nothing gives empty assignment
    assmt = StaticAssignment(NamedTuple(), NamedTuple())
    @test isempty(get_subassmt(assmt, :x))
    @test isempty(get_subassmt(assmt, :x => :y))

    # static_get_assmt on an address that contains nothing throws a KeyError
    assmt = StaticAssignment(NamedTuple(), NamedTuple())
    threw = false
    try static_get_subassmt(assmt, Val(:x)) catch KeyError threw = true end
    @test threw

    # get_value on an address that contains a subassmt throws a KeyError
    subassmt = DynamicAssignment()
    subassmt[:y] = 1
    assmt = StaticAssignment(NamedTuple(), (x=subassmt,))
    threw = false
    try get_value(assmt, :x) catch KeyError threw = true end
    @test threw

    # static_get_value on an address that contains a subassmt throws a KeyError
    subassmt = DynamicAssignment()
    subassmt[:y] = 1
    assmt = StaticAssignment(NamedTuple(), (x=subassmt,))
    threw = false
    try static_get_value(assmt, Val(:x)) catch KeyError threw = true end
    @test threw

    # get_value on an address that contains nothing throws a KeyError
    assmt = StaticAssignment(NamedTuple(), NamedTuple())
    threw = false
    try get_value(assmt, :x) catch KeyError threw = true end
    @test threw
    threw = false
    try get_value(assmt, :x => :y) catch KeyError threw = true end
    @test threw

    # static_get_value on an address that contains nothing throws a KeyError
    assmt = StaticAssignment(NamedTuple(), NamedTuple())
    threw = false
    try static_get_value(assmt, Val(:x)) catch KeyError threw = true end
    @test threw
end

@testset "dynamic assignment errors" begin

    # get_assmt on an address that contains a value throws a KeyError
    assmt = DynamicAssignment()
    assmt[:x] = 1
    threw = false
    try get_subassmt(assmt, :x) catch KeyError threw = true end
    @test threw

    # get_assmt on an address whose prefix contains a value throws a KeyError
    assmt = DynamicAssignment()
    assmt[:x] = 1
    threw = false
    try get_subassmt(assmt, :x => :y) catch KeyError threw = true end
    @test threw

    # get_assmt on an address that contains nothing gives empty assignment
    assmt = DynamicAssignment()
    @test isempty(get_subassmt(assmt, :x))
    @test isempty(get_subassmt(assmt, :x => :y))

    # get_value on an address that contains a subassmt throws a KeyError
    assmt = DynamicAssignment()
    assmt[:x => :y] = 1
    threw = false
    try get_value(assmt, :x) catch KeyError threw = true end
    @test threw

    # get_value on an address that contains nothing throws a KeyError
    assmt = DynamicAssignment()
    threw = false
    try get_value(assmt, :x) catch KeyError threw = true end
    @test threw
    threw = false
    try get_value(assmt, :x => :y) catch KeyError threw = true end
    @test threw
end

@testset "dynamic assignment overwrite" begin

    # overwrite value with a value
    assmt = DynamicAssignment()
    assmt[:x] = 1
    assmt[:x] = 2
    @test assmt[:x] == 2

    # overwrite value with a subassmt
    assmt = DynamicAssignment()
    assmt[:x] = 1
    subassmt = DynamicAssignment(); subassmt[:y] = 2
    set_subassmt!(assmt, :x, subassmt)
    @test !has_value(assmt, :x)
    @test !isempty(get_subassmt(assmt, :x))

    # overwrite subassignment with a value
    assmt = DynamicAssignment()
    assmt[:x => :y] = 1
    assmt[:x] = 2
    threw = false
    try get_subassmt(assmt, :x) catch KeyError threw = true end
    @test threw
    @test assmt[:x] == 2

    # overwrite subassignment with a subassignment
    assmt = DynamicAssignment()
    assmt[:x => :y] = 1
    subassmt = DynamicAssignment(); subassmt[:z] = 2
    set_subassmt!(assmt, :x,  subassmt)
    @test !isempty(get_subassmt(assmt, :x))
    @test !has_value(assmt, :x => :y)
    @test assmt[:x => :z] == 2

    # illegal set value under existing value
    assmt = DynamicAssignment()
    assmt[:x] = 1
    threw = false
    try set_value!(assmt, :x => :y, 2) catch KeyError threw = true end
    @test threw

    # illegal set subassmt under existing value
    assmt = DynamicAssignment()
    assmt[:x] = 1
    subassmt = DynamicAssignment(); assmt[:z] = 2
    threw = false
    try set_subassmt!(assmt, :x => :y, subassmt) catch KeyError threw = true end
    @test threw
end

@testset "address_set" begin

    assmt = DynamicAssignment()
    assmt[:x] = 1
    assmt[:y => :a] = 2
    assmt[:y => :b] = 3
    assmt[:y => :c => :z] = 4

    set = address_set(assmt)
    @test has_leaf_node(set, :x)
    @test has_leaf_node(set, :y => :a)
    @test has_leaf_node(set, :y => :b)
    @test has_leaf_node(set, :y => :c => :z)
end

@testset "dynamic assignment constructor" begin

    assmt = DynamicAssignment((:x, 1), (:y => :a, 2), (:y => :b, 3))
    @test assmt[:x] == 1
    @test assmt[:y => :a] == 2
    @test assmt[:y => :b] == 3
end
