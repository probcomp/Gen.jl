@gen function f_dynamic(x, y)
    z ~ normal(0, 1)
    return x + y + z
end
@gen (static) function f_static(x, y)
    z ~ normal(0, 1)
    return x + y + z
end

for (lang, f) in [:dynamic => f_dynamic,
                  :static  => f_static]
    @testset "update(...) shorthand assuming unchanged args ($lang modeling lang)" begin
        trace0 = simulate(f, (5, 6))

        constraints = choicemap((:z, 0))
        trace1, _, _, discard = update(trace0, constraints)
        # The main test is that the shorthand version runs without crashing,
        # which is already shown by the time we get here.  Beyond that, let's
        # sanity-check that `update` did what it's supposed to.
        @test get_args(trace1) == (5, 6)
        @test trace1[:z] == 0
        @test :z in map(first, get_values_shallow(discard))
    end

    @testset "regenerate(...) shorthand assuming unchanged args ($lang modeling lang)" begin
        trace0 = simulate(f, (5, 6))
        trace1, _, _ = regenerate(trace0, select(:z))
        # The main test is that the shorthand version runs without crashing,
        # which is already shown by the time we get here.  Beyond that, let's
        # sanity-check that `regenerate` did what it's supposed to.
        @test get_args(trace1) == (5, 6)
        @test trace1[:z] != trace0[:z]
    end
end
