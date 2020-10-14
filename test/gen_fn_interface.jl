f_dynamic = (@gen function f(x, y)
                 z ~ normal(0, 1)
                 return x + y + z
             end)
f_static  = (@gen (static) function f(x, y)
                 z ~ normal(0, 1)
                 return x + y + z
             end)
@load_generated_functions()
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
        @test :z in keys(get_values_shallow(discard))
    end

    @testset "regenerate(...) shorthand assuming unchanged args ($lang modeling lang)" begin
        @gen function f(x, y)
            z ~ normal(0, 1)
            return x + y + z
        end
        trace0 = simulate(f, (5, 6))

        trace1, _, _ = regenerate(trace0, select(:z))
        # The main test is that the shorthand version runs without crashing,
        # which is already shown by the time we get here.  Beyond that, let's
        # sanity-check that `regenerate` did what it's supposed to.
        @test get_args(trace1) == (5, 6)
        @test trace1[:z] != trace0[:z]
    end
end

