@testset "static assignment to/from array" begin
    submap = StaticChoiceMap((a=1., b=[2., 2.5]),NamedTuple())
    outer = StaticChoiceMap((c=3.,), (d=submap, e=submap))

    arr = to_array(outer, Float64)
    @test to_array(outer, Float64) == Float64[3.0, 1.0, 2.0, 2.5, 1.0, 2.0, 2.5]

    choices = from_array(outer, Float64[1, 2, 3, 4, 5, 6, 7])
    @test choices[:c] == 1.0
    @test choices[:d => :a] == 2.0
    @test choices[:d => :b] == [3.0, 4.0]
    @test choices[:e => :a] == 5.0
    @test choices[:e => :b] == [6.0, 7.0]
    @test length(collect(get_submaps_shallow(choices))) == 2
    @test length(collect(get_values_shallow(choices))) == 1
    submap1 = get_submap(choices, :d)
    @test length(collect(get_values_shallow(submap1))) == 2
    @test length(collect(get_submaps_shallow(submap1))) == 0
    submap2 = get_submap(choices, :e)
    @test length(collect(get_values_shallow(submap2))) == 2
    @test length(collect(get_submaps_shallow(submap2))) == 0
end

@testset "dynamic assignment to/from array" begin
    outer = choicemap()
    set_value!(outer, :c, 3.)
    submap = choicemap()
    set_value!(submap, :a, 1.)
    set_value!(submap, :b, [2., 2.5])
    set_submap!(outer, :d, submap)
    set_submap!(outer, :e, submap)

    arr = to_array(outer, Float64)
    @test to_array(outer, Float64) == Float64[3.0, 1.0, 2.0, 2.5, 1.0, 2.0, 2.5]

    choices = from_array(outer, Float64[1, 2, 3, 4, 5, 6, 7])
    @test choices[:c] == 1.0
    @test choices[:d => :a] == 2.0
    @test choices[:d => :b] == [3.0, 4.0]
    @test choices[:e => :a] == 5.0
    @test choices[:e => :b] == [6.0, 7.0]
    @test length(collect(get_submaps_shallow(choices))) == 2
    @test length(collect(get_values_shallow(choices))) == 1
    submap1 = get_submap(choices, :d)
    @test length(collect(get_values_shallow(submap1))) == 2
    @test length(collect(get_submaps_shallow(submap1))) == 0
    submap2 = get_submap(choices, :e)
    @test length(collect(get_values_shallow(submap2))) == 2
    @test length(collect(get_submaps_shallow(submap2))) == 0
end

@testset "dynamic assignment copy constructor" begin
    other = choicemap()
    other[:x] = 1
    other[:y] = 2
    other_nested = choicemap()
    other_nested[:z] = 3
    other_nested[:w] = 4
    set_submap!(other, :u, other_nested)
    choices = DynamicChoiceMap(other)
    @test choices[:x] == 1
    @test choices[:y] == 2
    @test choices[:u => :z] == 3
    @test choices[:u => :w] == 4
end

@testset "internal vector assignment to/from array" begin
    inner = choicemap()
    set_value!(inner, :a, 1.)
    set_value!(inner, :b, 2.)
    outer = vectorize_internal([inner, inner, inner])

    arr = to_array(outer, Float64)
    @test to_array(outer, Float64) == Float64[1, 2, 1, 2, 1, 2]

    choices = from_array(outer, Float64[1, 2, 3, 4, 5, 6])
    @test choices[1 => :a] == 1.0
    @test choices[1 => :b] == 2.0
    @test choices[2 => :a] == 3.0
    @test choices[2 => :b] == 4.0
    @test choices[3 => :a] == 5.0
    @test choices[3 => :b] == 6.0
    @test length(collect(get_submaps_shallow(choices))) == 3
end

@testset "dynamic assignment merge" begin
    submap = choicemap()
    set_value!(submap, :x, 1)
    choices1 = choicemap()
    set_value!(choices1, :a, 1.)
    set_value!(choices1, :b, 2.)
    set_submap!(choices1, :c, submap)
    set_submap!(choices1, :shared, submap)
    choices2 = choicemap()
    set_value!(choices2, :d, 3.)
    set_submap!(choices2, :e, submap)
    set_submap!(choices2, :f, submap)
    submap2 = choicemap()
    set_value!(submap2, :y, 4.)
    set_submap!(choices2, :shared, submap2)
    choices = merge(choices1, choices2)
    @test choices[:a] == 1.
    @test choices[:b] == 2.
    @test choices[:d] == 3.
    @test choices[:c => :x] == 1
    @test choices[:e => :x] == 1
    @test choices[:f => :x] == 1
    @test choices[:shared => :x] == 1
    @test choices[:shared => :y] == 4.
    @test length(collect(get_submaps_shallow(choices))) == 4
    @test length(collect(get_values_shallow(choices))) == 3
end

@testset "dynamic assignment variadic merge" begin
    choices1 = choicemap((:a, 1))
    choices2 = choicemap((:b, 2))
    choices3 = choicemap((:c, 3))
    choices_all = choicemap((:a, 1), (:b, 2), (:c, 3))
    @test merge(choices1) == choices1
    @test merge(choices1, choices2, choices3) == choices_all
end

@testset "static assignment merge" begin
    submap = choicemap()
    set_value!(submap, :x, 1)
    submap2 = choicemap()
    set_value!(submap2, :y, 4.)
    choices1 = StaticChoiceMap((a=1., b=2.), (c=submap, shared=submap))
    choices2 = StaticChoiceMap((d=3.,), (e=submap, f=submap, shared=submap2))
    choices = merge(choices1, choices2)
    @test choices[:a] == 1.
    @test choices[:b] == 2.
    @test choices[:d] == 3.
    @test choices[:c => :x] == 1
    @test choices[:e => :x] == 1
    @test choices[:f => :x] == 1
    @test choices[:shared => :x] == 1
    @test choices[:shared => :y] == 4.
    @test length(collect(get_submaps_shallow(choices))) == 4
    @test length(collect(get_values_shallow(choices))) == 3
end

@testset "static assignment variadic merge" begin
    choices1 = StaticChoiceMap((a=1,), NamedTuple())
    choices2 = StaticChoiceMap((b=2,), NamedTuple())
    choices3 = StaticChoiceMap((c=3,), NamedTuple())
    choices_all = StaticChoiceMap((a=1, b=2, c=3), NamedTuple())
    @test merge(choices1) == choices1
    @test merge(choices1, choices2, choices3) == choices_all
end

@testset "static assignment errors" begin

    # get_choices on an address that contains a value throws a KeyError
    choices = StaticChoiceMap((x=1,), NamedTuple())
    threw = false
    try get_submap(choices, :x) catch KeyError threw = true end
    @test threw

    # static_get_submap on an address that contains a value throws a KeyError
    choices = StaticChoiceMap((x=1,), NamedTuple())
    threw = false
    try static_get_submap(choices, Val(:x)) catch KeyError threw = true end
    @test threw

    # get_choices on an address whose prefix contains a value throws a KeyError
    choices = StaticChoiceMap((x=1,), NamedTuple())
    threw = false
    try get_submap(choices, :x => :y) catch KeyError threw = true end
    @test threw

    # static_get_choices on an address whose prefix contains a value throws a KeyError
    choices = StaticChoiceMap((x=1,), NamedTuple())
    threw = false
    try static_get_submap(choices, Val(:x)) catch KeyError threw = true end
    @test threw

    # get_choices on an address that contains nothing gives empty assignment
    choices = StaticChoiceMap(NamedTuple(), NamedTuple())
    @test isempty(get_submap(choices, :x))
    @test isempty(get_submap(choices, :x => :y))

    # static_get_choices on an address that contains nothing throws a KeyError
    choices = StaticChoiceMap(NamedTuple(), NamedTuple())
    threw = false
    try static_get_submap(choices, Val(:x)) catch KeyError threw = true end
    @test threw

    # get_value on an address that contains a submap throws a KeyError
    submap = choicemap()
    submap[:y] = 1
    choices = StaticChoiceMap(NamedTuple(), (x=submap,))
    threw = false
    try get_value(choices, :x) catch KeyError threw = true end
    @test threw

    # static_get_value on an address that contains a submap throws a KeyError
    submap = choicemap()
    submap[:y] = 1
    choices = StaticChoiceMap(NamedTuple(), (x=submap,))
    threw = false
    try static_get_value(choices, Val(:x)) catch KeyError threw = true end
    @test threw

    # get_value on an address that contains nothing throws a KeyError
    choices = StaticChoiceMap(NamedTuple(), NamedTuple())
    threw = false
    try get_value(choices, :x) catch KeyError threw = true end
    @test threw
    threw = false
    try get_value(choices, :x => :y) catch KeyError threw = true end
    @test threw

    # static_get_value on an address that contains nothing throws a KeyError
    choices = StaticChoiceMap(NamedTuple(), NamedTuple())
    threw = false
    try static_get_value(choices, Val(:x)) catch KeyError threw = true end
    @test threw
end

@testset "dynamic assignment errors" begin

    # get_choices on an address that contains a value throws a KeyError
    choices = choicemap()
    choices[:x] = 1
    threw = false
    try get_submap(choices, :x) catch KeyError threw = true end
    @test threw

    # get_choices on an address whose prefix contains a value throws a KeyError
    choices = choicemap()
    choices[:x] = 1
    threw = false
    try get_submap(choices, :x => :y) catch KeyError threw = true end
    @test threw

    # get_choices on an address that contains nothing gives empty assignment
    choices = choicemap()
    @test isempty(get_submap(choices, :x))
    @test isempty(get_submap(choices, :x => :y))

    # get_value on an address that contains a submap throws a KeyError
    choices = choicemap()
    choices[:x => :y] = 1
    threw = false
    try get_value(choices, :x) catch KeyError threw = true end
    @test threw

    # get_value on an address that contains nothing throws a KeyError
    choices = choicemap()
    threw = false
    try get_value(choices, :x) catch KeyError threw = true end
    @test threw
    threw = false
    try get_value(choices, :x => :y) catch KeyError threw = true end
    @test threw
end

@testset "dynamic assignment overwrite" begin

    # overwrite value with a value
    choices = choicemap()
    choices[:x] = 1
    choices[:x] = 2
    @test choices[:x] == 2

    # overwrite value with a submap
    choices = choicemap()
    choices[:x] = 1
    submap = choicemap(); submap[:y] = 2
    set_submap!(choices, :x, submap)
    @test !has_value(choices, :x)
    @test !isempty(get_submap(choices, :x))

    # overwrite subassignment with a value
    choices = choicemap()
    choices[:x => :y] = 1
    choices[:x] = 2
    threw = false
    try get_submap(choices, :x) catch KeyError threw = true end
    @test threw
    @test choices[:x] == 2

    # overwrite subassignment with a subassignment
    choices = choicemap()
    choices[:x => :y] = 1
    submap = choicemap(); submap[:z] = 2
    set_submap!(choices, :x,  submap)
    @test !isempty(get_submap(choices, :x))
    @test !has_value(choices, :x => :y)
    @test choices[:x => :z] == 2

    # illegal set value under existing value
    choices = choicemap()
    choices[:x] = 1
    threw = false
    try set_value!(choices, :x => :y, 2) catch KeyError threw = true end
    @test threw

    # illegal set submap under existing value
    choices = choicemap()
    choices[:x] = 1
    submap = choicemap(); choices[:z] = 2
    threw = false
    try set_submap!(choices, :x => :y, submap) catch KeyError threw = true end
    @test threw
end

@testset "dynamic assignment constructor" begin

    choices = choicemap((:x, 1), (:y => :a, 2))
    @test choices[:x] == 1
    @test choices[:y => :a] == 2
end

@testset "choice map equality" begin

    a = choicemap((:a, 1), (:b => :c => 2), (:d => :e => :f => 3))
    b = choicemap((:a, 1), (:d => :e => :f => 3))
    c = choicemap((:a, 1), (:b => :c => 2), (:d => :e => :f => 4))

    @test a == a
    @test b == b
    @test a != b
    @test b != a
    @test a != c
    @test c != a

end

@testset "choice map nested view" begin
    c = choicemap((:a, 1),
                  (:b => :c, 2))
    cv = nested_view(c)
    @test c[:a] == cv[:a]
    @test c[:b => :c] == cv[:b][:c]
    # Base.length
    @test length(cv) == 2
    @test length(cv[:b]) == 1
    # Base.keys
    @test Set(keys(cv)) == Set([:a, :b])
    @test Set(keys(cv[:b])) == Set([:c])
    # Base.:(==)
    c1 = choicemap((:a, 1),
                   (:b => :c, 2))
    c2 = choicemap((:a, 4),
                   (:b => :c, 2))
    @test nested_view(c1) == cv
    @test nested_view(c2) != cv
end

@testset "filtering choicemaps with selections" begin

    c = choicemap((:a, 1), (:b, 2))

    filtered = get_selected(c, select(:a))
    @test filtered[:a] == 1
    @test !has_value(filtered, :b)

    c = choicemap((:x => :y, 1), (:x => :z, 2))

    filtered = get_selected(c, select(:x))
    @test filtered[:x => :y] == 1
    @test filtered[:x => :z] == 2

    filtered = get_selected(c, select(:x => :y))
    @test filtered[:x => :y] == 1
    @test !has_value(filtered, :x => :z)
end
