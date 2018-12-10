@testset "map combinator" begin

    @gen function foo(x::Float64, y::Float64)
        @param std::Float64
        return @addr(normal(x + y, std), :z)
    end

    set_param!(foo, :std, 1.)

    @gen function bar(n::Int)
        xs = [1.0, 2.0, 3.0]
        ys = [3.0, 4.0, 5.0]
        return @addr(Map(foo)(xs[1:n], ys[1:n]), :map)
    end

    @testset "initialize" begin
        z1, z2 = 1.1, 2.2
        constraints = DynamicAssignment()
        constraints[:map => 1 => :z] = z1
        constraints[:map => 2 => :z] = z2
        (trace, weight) = initialize(bar, (2,), constraints)
        assignment = get_assignment(trace)
        @test assignment[:map => 1 => :z] == z1
        @test assignment[:map => 2 => :z] == z2
        @test isapprox(weight, logpdf(normal, z1, 4., 1.) + logpdf(normal, z2, 6., 1.))
    end

    @testset "propose" begin
        (assmt, weight) = propose(bar, (2,))
        z1 = assmt[:map => 1 => :z]
        z2 = assmt[:map => 2 => :z]
        @test isapprox(weight, logpdf(normal, z1, 4., 1.) + logpdf(normal, z2, 6., 1.))
    end

    @testset "assess" begin
        z1, z2 = 1.1, 2.2
        constraints = DynamicAssignment()
        constraints[:map => 1 => :z] = z1
        constraints[:map => 2 => :z] = z2
        (weight, retval) = assess(bar, (2,), constraints)
        @test length(retval) == 2
        @test isapprox(weight, logpdf(normal, z1, 4., 1.) + logpdf(normal, z2, 6., 1.))
    end

    @testset "force update" begin
        z1, z2 = 1.1, 2.2

        function get_initial_trace()
            constraints = DynamicAssignment()
            constraints[:map => 1 => :z] = z1
            constraints[:map => 2 => :z] = z2
            (trace, _) = initialize(bar, (2,), constraints)
            trace
        end

        # increasing length from 2 to 3 and change 2
        trace = get_initial_trace()
        z2_new = 3.3
        z3_new = 4.4
        constraints = DynamicAssignment()
        constraints[:map => 2 => :z] = z2_new
        constraints[:map => 3 => :z] = z3_new
        (trace, weight, discard, retdiff) = force_update((3,), nothing, trace, constraints)
        assignment = get_assignment(trace)
        @test get_args(trace) == (3,)
        @test assignment[:map => 1 => :z] == z1
        @test assignment[:map => 2 => :z] == z2_new
        @test assignment[:map => 3 => :z] == z3_new
        @test discard[:map => 2 => :z] == z2
        @test isapprox(get_score(trace), logpdf(normal, z1, 4., 1.) + logpdf(normal, z2_new, 6., 1.) + logpdf(normal, z3_new, 8., 1.))
        @test isapprox(weight, logpdf(normal, z3_new, 8., 1.) + logpdf(normal, z2_new, 6., 1.) - logpdf(normal, z2, 6., 1.))

        # decreasing length from 2 to 1 and change 1
        trace = get_initial_trace()
        z1_new = 3.3
        constraints = DynamicAssignment()
        constraints[:map => 1 => :z] = z1_new
        (trace, weight, discard, retdiff) = force_update((1,), nothing, trace, constraints)
        assignment = get_assignment(trace)
        @test get_args(trace) == (1,)
        @test !has_value(assignment, :map => 2 => :z)
        @test !has_value(assignment, :map => 3 => :z)
        @test assignment[:map => 1 => :z] == z1_new
        @test discard[:map => 1 => :z] == z1
        @test discard[:map => 2 => :z] == z2
        @test isapprox(get_score(trace), logpdf(normal, z1_new, 4., 1.))
        @test isapprox(weight, logpdf(normal, z1_new, 4., 1.) - logpdf(normal, z1, 4., 1.) - logpdf(normal, z2, 6., 1.))
    end

    @testset "fix update" begin
        z1, z2 = 1.1, 2.2

        function get_initial_trace()
            constraints = DynamicAssignment()
            constraints[:map => 1 => :z] = z1
            constraints[:map => 2 => :z] = z2
            (trace, _) = initialize(bar, (2,), constraints)
            trace
        end

        # increasing length from 2 to 3 and change 2
        trace = get_initial_trace()
        z2_new = 3.3
        constraints = DynamicAssignment()
        constraints[:map => 2 => :z] = z2_new
        (trace, weight, discard, retdiff) = fix_update((3,), nothing, trace, constraints)
        assignment = get_assignment(trace)
        @test get_args(trace) == (3,)
        @test assignment[:map => 1 => :z] == z1
        @test assignment[:map => 2 => :z] == z2_new
        z3_new = assignment[:map => 3 => :z]
        @test discard[:map => 2 => :z] == z2
        score = get_score(trace)
        @test isapprox(score, logpdf(normal, z1, 4., 1.) + logpdf(normal, z2_new, 6., 1.) + logpdf(normal, z3_new, 8., 1.))
        @test isapprox(weight, logpdf(normal, z2_new, 6., 1.) - logpdf(normal, z2, 6., 1.))

        # decreasing length from 2 to 1 and change 1
        trace = get_initial_trace()
        z1_new = 3.3
        constraints = DynamicAssignment()
        constraints[:map => 1 => :z] = z1_new
        (trace, weight, discard, retdiff) = fix_update((1,), nothing, trace, constraints)
        assignment = get_assignment(trace)
        @test get_args(trace) == (1,)
        @test !has_value(assignment, :map => 2 => :z)
        @test !has_value(assignment, :map => 3 => :z)
        @test assignment[:map => 1 => :z] == z1_new
        @test discard[:map => 1 => :z] == z1
        @test !has_value(discard, :map => 2 => :z)
        @test isapprox(get_score(trace), logpdf(normal, z1_new, 4., 1.))
        @test isapprox(weight, logpdf(normal, z1_new, 4., 1.) - logpdf(normal, z1, 4., 1.))
    end

    @testset "free update" begin
        z1, z2 = 1.1, 2.2

        function get_initial_trace()
            constraints = DynamicAssignment()
            constraints[:map => 1 => :z] = z1
            constraints[:map => 2 => :z] = z2
            (trace, _) = initialize(bar, (2,), constraints)
            trace
        end

        # increasing length from 2 to 3 and change 2
        trace = get_initial_trace()
        selection = DynamicAddressSet()
        push_leaf_node!(selection, :map => 2 => :z)
        (trace, weight, retdiff) = free_update((3,), nothing, trace, selection)
        assignment = get_assignment(trace)
        @test get_args(trace) == (3,)
        @test assignment[:map => 1 => :z] == z1
        z2_new = assignment[:map => 2 => :z]
        z3_new = assignment[:map => 3 => :z]
        score = get_score(trace)
        @test isapprox(score, logpdf(normal, z1, 4., 1.) + logpdf(normal, z2_new, 6., 1.) + logpdf(normal, z3_new, 8., 1.))
        @test isapprox(weight, 0.)

        # decreasing length from 2 to 1 and change 1
        trace = get_initial_trace()
        selection = DynamicAddressSet()
        push_leaf_node!(selection, :map => 1 => :z)
        (trace, weight, retdiff) = free_update((1,), nothing, trace, selection)
        assignment = get_assignment(trace)
        @test get_args(trace) == (1,)
        @test !has_value(assignment, :map => 2 => :z)
        @test !has_value(assignment, :map => 3 => :z)
        z1_new = assignment[:map => 1 => :z]
        @test isapprox(get_score(trace), logpdf(normal, z1_new, 4., 1.))
        @test isapprox(weight, 0.)
    end
end
