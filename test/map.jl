@testset "map combinator" begin
    
    # TODO test retdiffs

    @gen function foo(@grad(x::Float64), @grad(y::Float64))
        @param std::Float64
        return @addr(normal(x + y, std), :z)
    end

    set_param!(foo, :std, 1.)

    @gen function bar(n::Int)
        @param xs::Vector{Float64}
        @param ys::Vector{Float64}
        return @addr(Map(foo)(xs[1:n], ys[1:n]), :map)
    end

    set_param!(bar, :xs, [1.0, 2.0, 3.0, 4.0])
    set_param!(bar, :ys, [3.0, 4.0, 5.0, 6.0])

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
        (trace, weight, discard, retdiff) = force_update((3,), unknownargdiff, trace, constraints)
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
        (trace, weight, discard, retdiff) = force_update((1,), unknownargdiff, trace, constraints)
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
        (trace, weight, discard, retdiff) = fix_update((3,), unknownargdiff, trace, constraints)
        assignment = get_assignment(trace)
        @test get_args(trace) == (3,)
        @test assignment[:map => 1 => :z] == z1
        @test assignment[:map => 2 => :z] == z2_new
        z3_new = assignment[:map => 3 => :z]
        @test discard[:map => 2 => :z] == z2
        score = get_score(trace)
        @test isapprox(score, (logpdf(normal, z1, 4., 1.)
            + logpdf(normal, z2_new, 6., 1.)
            + logpdf(normal, z3_new, 8., 1.)))
        @test isapprox(weight, logpdf(normal, z2_new, 6., 1.) - logpdf(normal, z2, 6., 1.))

        # decreasing length from 2 to 1 and change 1
        trace = get_initial_trace()
        z1_new = 3.3
        constraints = DynamicAssignment()
        constraints[:map => 1 => :z] = z1_new
        (trace, weight, discard, retdiff) = fix_update((1,), unknownargdiff, trace, constraints)
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
        (trace, weight, retdiff) = free_update((3,), unknownargdiff, trace, selection)
        assignment = get_assignment(trace)
        @test get_args(trace) == (3,)
        @test assignment[:map => 1 => :z] == z1
        z2_new = assignment[:map => 2 => :z]
        z3_new = assignment[:map => 3 => :z]
        score = get_score(trace)
        @test isapprox(score, (logpdf(normal, z1, 4., 1.)
            + logpdf(normal, z2_new, 6., 1.)
            + logpdf(normal, z3_new, 8., 1.)))
        @test isapprox(weight, 0.)

        # decreasing length from 2 to 1 and change 1
        trace = get_initial_trace()
        selection = DynamicAddressSet()
        push_leaf_node!(selection, :map => 1 => :z)
        (trace, weight, retdiff) = free_update((1,), unknownargdiff, trace, selection)
        assignment = get_assignment(trace)
        @test get_args(trace) == (1,)
        @test !has_value(assignment, :map => 2 => :z)
        @test !has_value(assignment, :map => 3 => :z)
        z1_new = assignment[:map => 1 => :z]
        @test isapprox(get_score(trace), logpdf(normal, z1_new, 4., 1.))
        @test isapprox(weight, 0.)
    end

    @testset "extend" begin
        z1, z2 = 1.1, 2.2

        function get_initial_trace()
            constraints = DynamicAssignment()
            constraints[:map => 1 => :z] = z1
            constraints[:map => 2 => :z] = z2
            (trace, _) = initialize(bar, (2,), constraints)
            trace
        end

        # increasing length from 2 to 4; constrain 4 and let 3 be generated from prior
        trace = get_initial_trace()
        z4 = 4.4
        constraints = DynamicAssignment()
        constraints[:map => 4 => :z] = z4
        (trace, weight, retdiff) = extend((4,), unknownargdiff, trace, constraints)
        assignment = get_assignment(trace)
        @test get_args(trace) == (4,)
        @test assignment[:map => 1 => :z] == z1
        @test assignment[:map => 2 => :z] == z2
        @test assignment[:map => 4 => :z] == z4
        z3 = assignment[:map => 3 => :z]
        score = get_score(trace)
        @test isapprox(score, (logpdf(normal, z1, 4., 1.)
            + logpdf(normal, z2, 6., 1.)
            + logpdf(normal, z3, 8., 1.)
            + logpdf(normal, z4, 10., 1)))
        @test isapprox(weight, logpdf(normal, z4 , 10., 1.))
    end

    @testset "backprop_trace" begin
        z1, z2 = 1.1, 2.2
        xs = [1.0, 2.0]
        ys = [3.0, 4.0]

        function get_initial_trace()
            constraints = DynamicAssignment()
            constraints[1 => :z] = z1
            constraints[2 => :z] = z2
            (trace, _) = initialize(Map(foo), (xs, ys), constraints)
            trace
        end

        retval_grad = rand(2)

        expected_xs_grad = [logpdf_grad(normal, z1, 4., 1.)[2], logpdf_grad(normal, z2, 6., 1.)[2]]
        expected_ys_grad = [logpdf_grad(normal, z1, 4., 1.)[2], logpdf_grad(normal, z2, 6., 1.)[2]]
        expected_z2_grad = logpdf_grad(normal, z2, 6., 1.)[1] + retval_grad[2]

        # get gradients wrt xs and ys, and wrt address ':map => 2 => :z'
        trace = get_initial_trace()
        selection = DynamicAddressSet()
        push_leaf_node!(selection, 2 => :z)
        (input_grads, value_assmt, gradient_assmt) = backprop_trace(trace, selection, retval_grad)
        @test isapprox(input_grads[1], expected_xs_grad)
        @test isapprox(input_grads[2], expected_ys_grad)
        @test !has_value(value_assmt, 1 => :z)
        @test value_assmt[2 => :z] == z2
        @test !has_value(gradient_assmt, 1 => :z)
        @test isapprox(gradient_assmt[2 => :z], expected_z2_grad)
    end
end
