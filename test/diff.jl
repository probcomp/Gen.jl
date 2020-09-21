@testset "diff arithmetic" begin

    @testset "+" begin
        a = Diffed(1, NoChange())
        b = Diffed(2, NoChange())
        @test get_diff(a + b) == NoChange()
        @test strip_diff(a + b) == 3

        a = 1
        b = Diffed(2, NoChange())
        @test get_diff(a + b) == NoChange()
        @test strip_diff(a + b) == 3

        a = Diffed(1, NoChange())
        b = 2
        @test get_diff(a + b) == NoChange()
        @test strip_diff(a + b) == 3

        a = Diffed(1, UnknownChange())
        b = Diffed(2, UnknownChange())
        @test get_diff(a + b) == UnknownChange()
        @test strip_diff(a + b) == 3

        a = 1
        b = Diffed(2, UnknownChange())
        @test get_diff(a + b) == UnknownChange()
        @test strip_diff(a + b) == 3

        a = Diffed(1, UnknownChange())
        b = 2
        @test get_diff(a + b) == UnknownChange()
        @test strip_diff(a + b) == 3
    end

    # TODO test other binary operators
end

@testset "diff vectors" begin

    @testset "length" begin

    v = Diffed([1, 2, 3], NoChange())
    @test strip_diff(length(v)) == 3
    @test get_diff(length(v)) == NoChange()

    v = Diffed([1, 2, 3], UnknownChange())
    @test strip_diff(length(v)) == 3
    @test get_diff(length(v)) == UnknownChange()

    v = Diffed([1, 2, 3], VectorDiff(3, 4, Dict{Int,Diff}()))
    @test strip_diff(length(v)) == 3
    @test get_diff(length(v)) == IntDiff(-1)

    end

    @testset "getindex" begin

    v = [1, 2, 3]
    i = Diffed(2, UnknownChange())
    @test strip_diff(v[i]) == 2
    @test get_diff(v[i]) == UnknownChange()

    v = [1, 2, 3]
    i = Diffed(2, NoChange())
    @test strip_diff(v[i]) == 2
    @test get_diff(v[i]) == NoChange()

    v = [1, 2, 3]
    i = Diffed(2, IntDiff(1))
    @test strip_diff(v[i]) == 2
    @test get_diff(v[i]) == UnknownChange()

    v = Diffed([1, 2, 3], NoChange())
    @test strip_diff(v[2]) == 2
    @test get_diff(v[2]) == NoChange()

    v = Diffed([1, 2, 3], UnknownChange())
    @test strip_diff(v[2]) == 2
    @test get_diff(v[2]) == UnknownChange()

    # the value at v[2] was not changed
    v = Diffed([1, 2, 3], VectorDiff(3, 4, Dict{Int,Diff}()))
    @test strip_diff(v[2]) == 2
    @test get_diff(v[2]) == NoChange()

    # the value at v[2] was updated (reduced by 3)
    element_diff = IntDiff(-3)
    v = Diffed([1, 2, 3], VectorDiff(3, 4, Dict{Int,Diff}(2 => element_diff)))
    @test strip_diff(v[2]) == 2
    @test get_diff(v[2]) == element_diff
    end
end

@testset "diff dictionaries" begin
end

@testset "diff sets" begin
end
