@testset "Value" begin
    vcm1 = Value(2)
    vcm2 = Value(2.)
    vcm3 = Value([1,2])
    @test vcm1 isa Value{Int}
    @test vcm2 isa Value{Float64}
    @test vcm3 isa Value{Vector{Int}}
    @test vcm1[] == 2
    @test vcm1[] == get_value(vcm1)

    @test !isempty(vcm1)
    @test has_value(vcm1)
    @test get_value(vcm1) == 2
    @test vcm1 == vcm2
    @test isempty(get_submaps_shallow(vcm1))
    @test isempty(get_values_shallow(vcm1))
    @test isempty(get_nonvalue_submaps_shallow(vcm1))
    @test to_array(vcm1, Int) == [2]
    @test from_array(vcm1, [4]) == Value(4)
    @test from_array(vcm3, [4, 5]) == Value([4, 5])
    @test_throws Exception merge(vcm1, vcm2)
    @test_throws Exception merge(vcm1, choicemap(:a, 5))
    @test merge(vcm1, EmptyChoiceMap()) == vcm1
    @test merge(EmptyChoiceMap(), vcm1) == vcm1
    @test get_submap(vcm1, :addr) == EmptyChoiceMap()
    @test_throws ChoiceMapGetValueError get_value(vcm1, :addr)
    @test !has_value(vcm1, :addr)
    @test isapprox(vcm2, Value(prevfloat(2.)))
    @test isapprox(vcm1, Value(prevfloat(2.)))
    @test nested_view(vcm1) == 2
end

@testset "static choicemap constructors" begin
    @test StaticChoiceMap((a=Value(5), b=Value(6))) == StaticChoiceMap(a=5, b=6)
    submap = StaticChoiceMap(a=1., b=[2., 2.5])
    @test submap == StaticChoiceMap((a=Value(1.), b=Value([2., 2.5])))
    outer = StaticChoiceMap(c=3, d=submap, e=submap)
    @test outer == StaticChoiceMap((c=Value(3), d=submap, e=submap))

    # what if we have an emptychoicemap?
    StaticChoiceMap(a=Value(5), b=choicemap((1, "hello")), c=EmptyChoiceMap())

    # convert dynamic --> static
    d1 = choicemap((:a, 1), (:b => :c, 2.), (:d, "yes"))
    s1 = StaticChoiceMap(d1)
    @test s1 == d1
    # should not be a deep conversion!
    @test get_subtree(s1, :b) === get_subtree(d1, :b)

    d2 = choicemap((:a, 3), (:b, -10.2))
    d3 = choicemap((1 => :a, 80), (2 => :a, 100), (3 => :a, 2))
    set_submap!(d2, :c, d3)
    s2 = StaticChoiceMap(d2)
    @test s2 == d2
    @test get_subtree(s2, :c) == d3
end

@testset "static assignment to/from array" begin
    submap = StaticChoiceMap(a=1., b=[2., 2.5])
    outer = StaticChoiceMap(c=3., d=submap, e=submap)
    
    arr = to_array(outer, Float64)
    @test to_array(outer, Float64) == Float64[3.0, 1.0, 2.0, 2.5, 1.0, 2.0, 2.5]
    
    choices = from_array(outer, Float64[1, 2, 3, 4, 5, 6, 7])
    @test choices[:c] == 1.0
    @test choices[:d => :a] == 2.0
    @test choices[:d => :b] == [3.0, 4.0]
    @test choices[:e => :a] == 5.0
    @test choices[:e => :b] == [6.0, 7.0]
    @test length(collect(get_submaps_shallow(choices))) == 3
    @test length(collect(get_nonvalue_submaps_shallow(choices))) == 2
    @test length(collect(get_values_shallow(choices))) == 1
    submap1 = get_submap(choices, :d)
    @test length(collect(get_values_shallow(submap1))) == 2
    @test length(collect(get_submaps_shallow(submap1))) == 2
    @test length(collect(get_nonvalue_submaps_shallow(submap1))) == 0
    submap2 = get_submap(choices, :e)
    @test length(collect(get_values_shallow(submap2))) == 2
    @test length(collect(get_nonvalue_submaps_shallow(submap2))) == 0
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
    @test get_submap(choices, :c) == Value(1.0)
    @test get_submap(choices, :d => :b) == Value([3.0, 4.0])
    @test length(collect(get_submaps_shallow(choices))) == 3
    @test length(collect(get_nonvalue_submaps_shallow(choices))) == 2
    @test length(collect(get_values_shallow(choices))) == 1
    submap1 = get_submap(choices, :d)
    @test length(collect(get_values_shallow(submap1))) == 2
    @test length(collect(get_submaps_shallow(submap1))) == 2
    @test length(collect(get_nonvalue_submaps_shallow(submap1))) == 0
    submap2 = get_submap(choices, :e)
    @test length(collect(get_values_shallow(submap2))) == 2
    @test length(collect(get_nonvalue_submaps_shallow(submap2))) == 0
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
    @test length(collect(get_nonvalue_submaps_shallow(choices))) == 4
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
    choices1 = StaticChoiceMap(a=1., b=2., c=submap, shared=submap)
    choices2 = StaticChoiceMap(d=3., e=submap, f=submap, shared=submap2)
    choices = merge(choices1, choices2)
    @test choices[:a] == 1.
    @test choices[:b] == 2.
    @test choices[:d] == 3.
    @test choices[:c => :x] == 1
    @test choices[:e => :x] == 1
    @test choices[:f => :x] == 1
    @test choices[:shared => :x] == 1
    @test choices[:shared => :y] == 4.
    @test length(collect(get_nonvalue_submaps_shallow(choices))) == 4
    @test length(collect(get_values_shallow(choices))) == 3
end

@testset "static assignment variadic merge" begin
    choices1 = StaticChoiceMap(a=1)
    choices2 = StaticChoiceMap(b=2)
    choices3 = StaticChoiceMap(c=3)
    choices_all = StaticChoiceMap(a=1, b=2, c=3)
    @test merge(choices1) == choices1
    @test merge(choices1, choices2, choices3) == choices_all
end

# TODO: in changing a lot of these to reflect the new behavior of choicemap,
# they are mostly not error checks, but instead checks for returning `EmptyChoiceMap`;
# should we relabel this testset?
@testset "static assignment errors" begin
    # get_choices on an address that returns a Value
    choices = StaticChoiceMap(x=1)
    @test get_submap(choices, :x) == Value(1)

    # static_get_submap on an address that contains a value returns a Value
    choices = StaticChoiceMap(x=1)
    @test static_get_submap(choices, Val(:x)) == Value(1)

    # get_submap on an address whose prefix contains a value returns EmptyChoiceMap
    choices = StaticChoiceMap(x=1)
    @test get_submap(choices, :x => :y) == EmptyChoiceMap()

    # get_choices on an address that contains nothing gives empty assignment
    choices = StaticChoiceMap()
    @test isempty(get_submap(choices, :x))
    @test isempty(get_submap(choices, :x => :y))

    # static_get_choices on an address that contains nothing returns an EmptyChoiceMap
    choices = StaticChoiceMap()
    @test static_get_submap(choices, Val(:x)) == EmptyChoiceMap()

    # get_value on an address that contains a submap throws a ChoiceMapGetValueError
    submap = choicemap()
    submap[:y] = 1
    choices = StaticChoiceMap(x=submap)
    @test_throws ChoiceMapGetValueError get_value(choices, :x)

    # static_get_value on an address that contains a submap throws a ChoiceMapGetValueError
    submap = choicemap()
    submap[:y] = 1
    choices = StaticChoiceMap(x=submap)
    @test_throws ChoiceMapGetValueError static_get_value(choices, Val(:x))

    # get_value on an address that contains nothing throws a ChoiceMapGetValueError
    choices = StaticChoiceMap()
    @test_throws ChoiceMapGetValueError get_value(choices, :x)
    @test_throws ChoiceMapGetValueError get_value(choices, :x => :y)

    # static_get_value on an address that contains nothing throws a ChoiceMapGetValueError
    choices = StaticChoiceMap()
    @test_throws ChoiceMapGetValueError static_get_value(choices, Val(:x))
end

@testset "dynamic assignment errors" begin
    # get_choices on an address that contains a value returns a Value
    choices = choicemap()
    choices[:x] = 1
    @test get_submap(choices, :x) == Value(1)

    # get_choices on an address whose prefix contains a value returns EmptyChoiceMap
    choices = choicemap()
    choices[:x] = 1
    @test get_submap(choices, :x => :y) == EmptyChoiceMap()

    # get_choices on an address that contains nothing gives empty assignment
    choices = choicemap()
    @test isempty(get_submap(choices, :x))
    @test isempty(get_submap(choices, :x => :y))

    # get_value on an address that contains a submap throws a ChoiceMapGetValueError
    choices = choicemap()
    choices[:x => :y] = 1
    @test_throws ChoiceMapGetValueError get_value(choices, :x)

    # get_value on an address that contains nothing throws a ChoiceMapGetValueError
    choices = choicemap()
    @test_throws ChoiceMapGetValueError get_value(choices, :x)
    @test_throws ChoiceMapGetValueError get_value(choices, :x => :y)
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
    @test get_submap(choices, :x) == Value(2)
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
    @test_throws Exception set_value!(choices, :x => :y, 2)

    # illegal set submap under existing value
    choices = choicemap()
    choices[:x] = 1
    submap = choicemap(); choices[:z] = 2
    @test_throws Exception set_submap!(choices, :x => :y, submap)
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
