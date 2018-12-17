@testset "unfold combinator" begin

    @gen function kernel(t::Int, x::Float64, @grad(alpha::Float64), @grad(beta::Float64))
        @param std::Float64
        x = @addr(normal(x * alpha + beta, std), :x)
        @diff begin
            xdiff = @choicediff(:x)
            @retdiff(isnodiff(xdiff) ? NoRetDiff() : DefaultRetDiff())
        end
        return x
    end
    
    std = 1.
    set_param!(kernel, :std, std)

    foo = Unfold(kernel)

    @testset "initialize" begin
        x_init = 0.1
        alpha = 0.2
        beta = 0.3
        x1 = 1.1
        x2 = 1.2
        x3 = 1.3
        constraints = DynamicAssignment()
        constraints[1 => :x] = x1
        constraints[3 => :x] = x3
        (trace, weight) = initialize(foo, (3, x_init, alpha, beta), constraints)
        assmt = get_assmt(trace)
        @test assmt[1 => :x] == x1
        @test assmt[3 => :x] == x3
        @test length(collect(get_values_shallow(assmt))) == 0
        @test length(collect(get_subassmts_shallow(assmt))) == 3
        x2 = assmt[2 => :x]
        expected_weight = (logpdf(normal, x1, x_init * alpha + beta, std)
             + logpdf(normal, x3, x2 * alpha + beta, std))
        expected_score = (logpdf(normal, x1, x_init * alpha + beta, std)
             + logpdf(normal, x2, x1 * alpha + beta, std)
             + logpdf(normal, x3, x2 * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        @test isapprox(get_score(trace), expected_score)
        retval = get_retval(trace)
        @test length(retval) == 3
        @test retval[1] == x1
        @test retval[2] == x2
        @test retval[3] == x3
    end

    @testset "propose" begin
        x_init = 0.1
        alpha = 0.2
        beta = 0.3
        (assmt, weight, retval) = propose(foo, (3, x_init, alpha, beta))
        @test length(collect(get_values_shallow(assmt))) == 0
        @test length(collect(get_subassmts_shallow(assmt))) == 3
        x1 = assmt[1 => :x]
        x2 = assmt[2 => :x]
        x3 = assmt[3 => :x]
        expected_weight = (logpdf(normal, x1, x_init * alpha + beta, std)
             + logpdf(normal, x2, x1 * alpha + beta, std)
             + logpdf(normal, x3, x2 * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        @test length(retval) == 3
        @test retval[1] == x1
        @test retval[2] == x2
        @test retval[3] == x3
    end

    @testset "assess" begin
        x_init = 0.1
        alpha = 0.2
        beta = 0.3
        x1 = 1.1
        x2 = 1.2
        x3 = 1.3
        assmt = DynamicAssignment()
        assmt[1 => :x] = x1
        assmt[2 => :x] = x2
        assmt[3 => :x] = x3
        (weight, retval) = assess(foo, (3, x_init, alpha, beta), assmt)
        expected_weight = (logpdf(normal, x1, x_init * alpha + beta, std)
             + logpdf(normal, x2, x1 * alpha + beta, std)
             + logpdf(normal, x3, x2 * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        @test length(retval) == 3
        @test retval[1] == x1
        @test retval[2] == x2
        @test retval[3] == x3
    end

    @testset "force update" begin
        x_init = 0.1
        alpha = 0.2
        beta = 0.3
        x1 = 1.1
        x2 = 1.2

        function get_initial_trace()
            constraints = DynamicAssignment()
            constraints[1 => :x] = x1
            constraints[2 => :x] = x2
            (trace, _) = initialize(foo, (2, x_init, alpha, beta), constraints)
            trace
        end

        # unknownargdiff, increasing length from 2 to 3 and change 2 and change params
        trace = get_initial_trace()
        x2_new = 1.3
        x3_new = 1.4
        alpha_new = 0.5
        constraints = DynamicAssignment()
        constraints[2 => :x] = x2_new
        constraints[3 => :x] = x3_new
        (trace, weight, discard, retdiff) = force_update((3, x_init, alpha_new, beta),
            unknownargdiff, trace, constraints)
        assmt = get_assmt(trace)
        @test get_args(trace) == (3, x_init, alpha_new, beta)
        @test assmt[1 => :x] == x1
        @test assmt[2 => :x] == x2_new
        @test assmt[3 => :x] == x3_new
        @test discard[2 => :x] == x2
        expected_score = (logpdf(normal, x1, x_init * alpha_new + beta, std)
            + logpdf(normal, x2_new, x1 * alpha_new + beta, std)
            + logpdf(normal, x3_new, x2_new * alpha_new + beta, std))
        @test isapprox(get_score(trace), expected_score)
        # TODO changed, the effect of alpha_new on x1 was not considered
        expected_weight = (logpdf(normal, x3_new, x2_new * alpha_new + beta, std)
            + logpdf(normal, x2_new, x1 * alpha_new + beta, std)
            - logpdf(normal, x2, x1 * alpha + beta, std)
            + logpdf(normal, x1, x_init * alpha_new + beta, std)
            - logpdf(normal, x1, x_init * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        retval = get_retval(trace)
        @test length(retval) == 3
        @test retval[1] == x1
        @test retval[2] == x2_new
        @test retval[3] == x3_new
        @test isa(retdiff, VectorCustomRetDiff)
        @test !haskey(retdiff.retained_retdiffs, 1) # no diff
        @test retdiff.retained_retdiffs[2] == DefaultRetDiff() # retval changed
        @test !haskey(retdiff.retained_retdiffs, 3) # new, not retained
        @test !isnodiff(retdiff)

        # unknownargdiff, decreasing length from 2 to 1 and change 1 and change params
        trace = get_initial_trace()
        x1_new = 1.3
        alpha_new = 0.5
        constraints = DynamicAssignment()
        constraints[1 => :x] = x1_new
        (trace, weight, discard, retdiff) = force_update((1, x_init, alpha_new, beta),
            unknownargdiff, trace, constraints)
        assmt = get_assmt(trace)
        @test get_args(trace) == (1, x_init, alpha_new, beta)
        @test !has_value(assmt, 2 => :x)
        @test !has_value(assmt, 3 => :x)
        @test assmt[1 => :x] == x1_new
        @test discard[1 => :x] == x1
        @test discard[2 => :x] == x2
        @test isapprox(get_score(trace), logpdf(normal, x1_new, x_init * alpha_new + beta, std))
        expected_weight = (logpdf(normal, x1_new, x_init * alpha_new + beta, std)
            - logpdf(normal, x1, x_init * alpha + beta, std)
            - logpdf(normal, x2, x1 * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        retval = get_retval(trace)
        @test length(retval) == 1
        @test retval[1] == x1_new
        @test isa(retdiff, VectorCustomRetDiff)
        @test retdiff.retained_retdiffs[1] == DefaultRetDiff() # retval changed
        @test !haskey(retdiff.retained_retdiffs, 2) # removed, not retained
        @test !isnodiff(retdiff)

        # noargdiff, change nothing
        trace = get_initial_trace()
        (trace, weight, discard, retdiff) = force_update((2, x_init, alpha, beta),
            noargdiff, trace, EmptyAssignment())
        assmt = get_assmt(trace)
        @test get_args(trace) == (2, x_init, alpha, beta)
        @test assmt[1 => :x] == x1
        @test assmt[2 => :x] == x2
        @test isempty(discard)
        expected_score = (logpdf(normal, x1, x_init * alpha + beta, std)
            + logpdf(normal, x2, x1 * alpha + beta, std))
        @test isapprox(get_score(trace), expected_score)
        @test isapprox(weight, 0.)
        retval = get_retval(trace)
        @test length(retval) == 2
        @test retval[1] == x1
        @test retval[2] == x2
        @test isa(retdiff, NoRetDiff)
        @test isnodiff(retdiff)

        # noargdiff, change x2
        trace = get_initial_trace()
        x2_new = 3.3
        constraints = DynamicAssignment()
        constraints[2 => :x] = x2_new
        (trace, weight, discard, retdiff) = force_update((2, x_init, alpha, beta),
            noargdiff, trace, constraints)
        assmt = get_assmt(trace)
        @test get_args(trace) == (2, x_init, alpha, beta)
        @test assmt[1 => :x] == x1
        @test assmt[2 => :x] == x2_new
        @test discard[2 => :x] == x2
        expected_score = (logpdf(normal, x1, x_init * alpha + beta, std)
            + logpdf(normal, x2_new, x1 * alpha + beta, std))
        @test isapprox(get_score(trace), expected_score)
        expected_weight = (logpdf(normal, x2_new, x1 * alpha + beta, std)
            - logpdf(normal, x2, x1 * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        retval = get_retval(trace)
        @test length(retval) == 2
        @test retval[1] == x1
        @test retval[2] == x2_new
        @test isa(retdiff, VectorCustomRetDiff)
        @test !haskey(retdiff.retained_retdiffs, 1) # no diff
        @test retdiff.retained_retdiffs[2] == DefaultRetDiff() # retval changed
        @test !isnodiff(retdiff)

        # init_changed=true, params_changed=false, change nothing
        trace = get_initial_trace()
        x_init_new = 0.1
        (trace, weight, discard, retdiff) = force_update((2, x_init_new, alpha, beta),
            UnfoldCustomArgDiff(true, false), trace, EmptyAssignment())
        assmt = get_assmt(trace)
        @test get_args(trace) == (2, x_init_new, alpha, beta)
        @test assmt[1 => :x] == x1
        @test assmt[2 => :x] == x2
        @test isempty(discard)
        expected_score = (logpdf(normal, x1, x_init_new * alpha + beta, std)
            + logpdf(normal, x2, x1 * alpha + beta, std))
        @test isapprox(get_score(trace), expected_score)
        expected_weight = (logpdf(normal, x1, x_init_new * alpha + beta, std)
            - logpdf(normal, x1, x_init * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        retval = get_retval(trace)
        @test length(retval) == 2
        @test retval[1] == x1
        @test retval[2] == x2
        @test isa(retdiff, NoRetDiff)
        @test isnodiff(retdiff)

        # init_changed=false, params_changed=true, change nothing
        trace = get_initial_trace()
        alpha_new = 0.5
        (trace, weight, discard, retdiff) = force_update((2, x_init, alpha_new, beta),
            UnfoldCustomArgDiff(false, true), trace, EmptyAssignment())
        assmt = get_assmt(trace)
        @test get_args(trace) == (2, x_init, alpha_new, beta)
        @test assmt[1 => :x] == x1
        @test assmt[2 => :x] == x2
        @test isempty(discard)
        expected_score = (logpdf(normal, x1, x_init * alpha_new + beta, std)
            + logpdf(normal, x2, x1 * alpha_new + beta, std))
        @test isapprox(get_score(trace), expected_score)
        expected_weight = (logpdf(normal, x1, x_init * alpha_new + beta, std)
            - logpdf(normal, x1, x_init * alpha + beta, std)
            + logpdf(normal, x2, x1 * alpha_new + beta, std)
            - logpdf(normal, x2, x1 * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        retval = get_retval(trace)
        @test length(retval) == 2
        @test retval[1] == x1
        @test retval[2] == x2
        @test isa(retdiff, NoRetDiff)
        @test isnodiff(retdiff)
    end

    @testset "fix update" begin
        x_init = 0.1
        alpha = 0.2
        beta = 0.3
        x1 = 1.1
        x2 = 1.2

        function get_initial_trace()
            constraints = DynamicAssignment()
            constraints[1 => :x] = x1
            constraints[2 => :x] = x2
            (trace, _) = initialize(foo, (2, x_init, alpha, beta), constraints)
            trace
        end

        # unknownargdiff, increasing length from 2 to 3 and change 2 and change params
        trace = get_initial_trace()
        x2_new = 1.3
        alpha_new = 0.5
        constraints = DynamicAssignment()
        constraints[2 => :x] = x2_new
        (trace, weight, discard, retdiff) = fix_update((3, x_init, alpha_new, beta),
            unknownargdiff, trace, constraints)
        assmt = get_assmt(trace)
        @test get_args(trace) == (3, x_init, alpha_new, beta)
        @test assmt[1 => :x] == x1
        @test assmt[2 => :x] == x2_new
        x3_new = assmt[3 => :x]
        @test discard[2 => :x] == x2
        expected_score = (logpdf(normal, x1, x_init * alpha_new + beta, std)
            + logpdf(normal, x2_new, x1 * alpha_new + beta, std)
            + logpdf(normal, x3_new, x2_new * alpha_new + beta, std))
        @test isapprox(get_score(trace), expected_score)
        expected_weight = (logpdf(normal, x1, x_init * alpha_new + beta, std)
            - logpdf(normal, x1, x_init * alpha + beta, std)
            + logpdf(normal, x2_new, x1 * alpha_new + beta, std)
            - logpdf(normal, x2, x1 * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        retval = get_retval(trace)
        @test length(retval) == 3
        @test retval[1] == x1
        @test retval[2] == x2_new
        @test retval[3] == x3_new
        @test isa(retdiff, VectorCustomRetDiff)
        @test !haskey(retdiff.retained_retdiffs, 1) # no diff
        @test retdiff.retained_retdiffs[2] == DefaultRetDiff() # retval changed
        @test !haskey(retdiff.retained_retdiffs, 3) # new, not retained
        @test !isnodiff(retdiff)

        # unknownargdiff, decreasing length from 2 to 1 and change 1 and change params
        trace = get_initial_trace()
        x1_new = 1.3
        alpha_new = 0.5
        constraints = DynamicAssignment()
        constraints[1 => :x] = x1_new
        (trace, weight, discard, retdiff) = fix_update((1, x_init, alpha_new, beta),
            unknownargdiff, trace, constraints)
        assmt = get_assmt(trace)
        @test get_args(trace) == (1, x_init, alpha_new, beta)
        @test !has_value(assmt, 2 => :x)
        @test !has_value(assmt, 3 => :x)
        @test assmt[1 => :x] == x1_new
        @test discard[1 => :x] == x1
        @test !has_value(discard, 3 => :x)
        @test isapprox(get_score(trace), logpdf(normal, x1_new, x_init * alpha_new + beta, std))
        expected_weight = (logpdf(normal, x1_new, x_init * alpha_new + beta, std)
            - logpdf(normal, x1, x_init * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        retval = get_retval(trace)
        @test length(retval) == 1
        @test retval[1] == x1_new
        @test isa(retdiff, VectorCustomRetDiff)
        @test retdiff.retained_retdiffs[1] == DefaultRetDiff() # retval changed
        @test !haskey(retdiff.retained_retdiffs, 2) # removed, not retained
        @test !isnodiff(retdiff)

        # noargdiff, change nothing
        trace = get_initial_trace()
        (trace, weight, discard, retdiff) = fix_update((2, x_init, alpha, beta),
            noargdiff, trace, EmptyAssignment())
        assmt = get_assmt(trace)
        @test get_args(trace) == (2, x_init, alpha, beta)
        @test assmt[1 => :x] == x1
        @test assmt[2 => :x] == x2
        @test isempty(discard)
        expected_score = (logpdf(normal, x1, x_init * alpha + beta, std)
            + logpdf(normal, x2, x1 * alpha + beta, std))
        @test isapprox(get_score(trace), expected_score)
        @test isapprox(weight, 0.)
        retval = get_retval(trace)
        @test length(retval) == 2
        @test retval[1] == x1
        @test retval[2] == x2
        @test isa(retdiff, NoRetDiff)
        @test isnodiff(retdiff)

        # noargdiff, change x2
        trace = get_initial_trace()
        x2_new = 3.3
        constraints = DynamicAssignment()
        constraints[2 => :x] = x2_new
        (trace, weight, discard, retdiff) = fix_update((2, x_init, alpha, beta),
            noargdiff, trace, constraints)
        assmt = get_assmt(trace)
        @test get_args(trace) == (2, x_init, alpha, beta)
        @test assmt[1 => :x] == x1
        @test assmt[2 => :x] == x2_new
        @test discard[2 => :x] == x2
        expected_score = (logpdf(normal, x1, x_init * alpha + beta, std)
            + logpdf(normal, x2_new, x1 * alpha + beta, std))
        @test isapprox(get_score(trace), expected_score)
        expected_weight = (logpdf(normal, x2_new, x1 * alpha + beta, std)
            - logpdf(normal, x2, x1 * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        retval = get_retval(trace)
        @test length(retval) == 2
        @test retval[1] == x1
        @test retval[2] == x2_new
        @test isa(retdiff, VectorCustomRetDiff)
        @test !haskey(retdiff.retained_retdiffs, 1) # no diff
        @test retdiff.retained_retdiffs[2] == DefaultRetDiff() # retval changed
        @test !isnodiff(retdiff)

        # init_changed=true, params_changed=false, change nothing
        trace = get_initial_trace()
        x_init_new = -0.1
        (trace, weight, discard, retdiff) = fix_update((2, x_init_new, alpha, beta),
            UnfoldCustomArgDiff(true, false), trace, EmptyAssignment())
        assmt = get_assmt(trace)
        @test get_args(trace) == (2, x_init_new, alpha, beta)
        @test assmt[1 => :x] == x1
        @test assmt[2 => :x] == x2
        @test isempty(discard)
        expected_score = (logpdf(normal, x1, x_init_new * alpha + beta, std)
            + logpdf(normal, x2, x1 * alpha + beta, std))
        @test isapprox(get_score(trace), expected_score)
        expected_weight = (logpdf(normal, x1, x_init_new * alpha + beta, std)
            - logpdf(normal, x1, x_init * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        retval = get_retval(trace)
        @test length(retval) == 2
        @test retval[1] == x1
        @test retval[2] == x2
        @test isa(retdiff, NoRetDiff)
        @test isnodiff(retdiff)

        # init_changed=false, params_changed=true, change nothing
        trace = get_initial_trace()
        alpha_new = 0.5
        (trace, weight, discard, retdiff) = fix_update((2, x_init, alpha_new, beta),
            UnfoldCustomArgDiff(false, true), trace, EmptyAssignment())
        assmt = get_assmt(trace)
        @test get_args(trace) == (2, x_init, alpha_new, beta)
        @test assmt[1 => :x] == x1
        @test assmt[2 => :x] == x2
        @test isempty(discard)
        expected_score = (logpdf(normal, x1, x_init * alpha_new + beta, std)
            + logpdf(normal, x2, x1 * alpha_new + beta, std))
        @test isapprox(get_score(trace), expected_score)
        expected_weight = (logpdf(normal, x1, x_init * alpha_new + beta, std)
            - logpdf(normal, x1, x_init * alpha + beta, std)
            + logpdf(normal, x2, x1 * alpha_new + beta, std)
            - logpdf(normal, x2, x1 * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        retval = get_retval(trace)
        @test length(retval) == 2
        @test retval[1] == x1
        @test retval[2] == x2
        @test isa(retdiff, NoRetDiff)
        @test isnodiff(retdiff)
    end

    @testset "free update" begin
        x_init = 0.1
        alpha = 0.2
        beta = 0.3
        x1 = 1.1
        x2 = 1.2

        function get_initial_trace()
            constraints = DynamicAssignment()
            constraints[1 => :x] = x1
            constraints[2 => :x] = x2
            (trace, _) = initialize(foo, (2, x_init, alpha, beta), constraints)
            trace
        end

        # unknownargdiff, increasing length from 2 to 3 and change 2 and change params
        trace = get_initial_trace()
        alpha_new = 0.5
        selection = DynamicAddressSet()
        push_leaf_node!(selection, 2 => :x)
        (trace, weight, retdiff) = free_update((3, x_init, alpha_new, beta),
            unknownargdiff, trace, selection)
        assmt = get_assmt(trace)
        @test get_args(trace) == (3, x_init, alpha_new, beta)
        @test assmt[1 => :x] == x1
        x2_new = assmt[2 => :x]
        x3_new = assmt[3 => :x]
        expected_score = (logpdf(normal, x1, x_init * alpha_new + beta, std)
            + logpdf(normal, x2_new, x1 * alpha_new + beta, std)
            + logpdf(normal, x3_new, x2_new * alpha_new + beta, std))
        @test isapprox(get_score(trace), expected_score)
        expected_weight = (logpdf(normal, x1, x_init * alpha_new + beta, std)
            - logpdf(normal, x1, x_init * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        retval = get_retval(trace)
        @test length(retval) == 3
        @test retval[1] == x1
        @test retval[2] == x2_new
        @test retval[3] == x3_new
        @test isa(retdiff, VectorCustomRetDiff)
        @test !haskey(retdiff.retained_retdiffs, 1) # no diff
        @test retdiff.retained_retdiffs[2] == DefaultRetDiff() # retval changed
        @test !haskey(retdiff.retained_retdiffs, 3) # new, not retained
        @test !isnodiff(retdiff)

        # unknownargdiff, decreasing length from 2 to 1 and change 1 and change params
        trace = get_initial_trace()
        alpha_new = 0.5
        selection = DynamicAddressSet()
        push_leaf_node!(selection, 1 => :x)
        (trace, weight, retdiff) = free_update((1, x_init, alpha_new, beta),
            unknownargdiff, trace, selection)
        assmt = get_assmt(trace)
        @test get_args(trace) == (1, x_init, alpha_new, beta)
        @test !has_value(assmt, 2 => :x)
        @test !has_value(assmt, 3 => :x)
        x1_new = assmt[1 => :x]
        @test isapprox(get_score(trace), logpdf(normal, x1_new, x_init * alpha_new + beta, std))
        @test isapprox(weight, 0.)
        retval = get_retval(trace)
        @test length(retval) == 1
        @test retval[1] == x1_new
        @test isa(retdiff, VectorCustomRetDiff)
        @test retdiff.retained_retdiffs[1] == DefaultRetDiff() # retval changed
        @test !haskey(retdiff.retained_retdiffs, 2) # removed, not retained
        @test !isnodiff(retdiff)

        # noargdiff, change nothing
        trace = get_initial_trace()
        (trace, weight, retdiff) = free_update((2, x_init, alpha, beta),
            noargdiff, trace, EmptyAddressSet())
        assmt = get_assmt(trace)
        @test get_args(trace) == (2, x_init, alpha, beta)
        @test assmt[1 => :x] == x1
        @test assmt[2 => :x] == x2
        expected_score = (logpdf(normal, x1, x_init * alpha + beta, std)
            + logpdf(normal, x2, x1 * alpha + beta, std))
        @test isapprox(get_score(trace), expected_score)
        @test isapprox(weight, 0.)
        retval = get_retval(trace)
        @test length(retval) == 2
        @test retval[1] == x1
        @test retval[2] == x2
        @test isa(retdiff, NoRetDiff)
        @test isnodiff(retdiff)

        # noargdiff, change x2
        trace = get_initial_trace()
        x2_new = 3.3
        selection = DynamicAddressSet()
        push_leaf_node!(selection, 2 => :x)
        (trace, weight, retdiff) = free_update((2, x_init, alpha, beta),
            noargdiff, trace, selection)
        assmt = get_assmt(trace)
        @test get_args(trace) == (2, x_init, alpha, beta)
        @test assmt[1 => :x] == x1
        x2_new = assmt[2 => :x]
        expected_score = (logpdf(normal, x1, x_init * alpha + beta, std)
            + logpdf(normal, x2_new, x1 * alpha + beta, std))
        @test isapprox(get_score(trace), expected_score)
        @test isapprox(weight, 0.)
        retval = get_retval(trace)
        @test length(retval) == 2
        @test retval[1] == x1
        @test retval[2] == x2_new
        @test isa(retdiff, VectorCustomRetDiff)
        @test !haskey(retdiff.retained_retdiffs, 1) # no diff
        @test retdiff.retained_retdiffs[2] == DefaultRetDiff() # retval changed
        @test !isnodiff(retdiff)

        # init_changed=true, params_changed=false, change nothing
        trace = get_initial_trace()
        x_init_new = -0.1
        (trace, weight, retdiff) = free_update((2, x_init_new, alpha, beta),
            UnfoldCustomArgDiff(true, false), trace, EmptyAddressSet())
        assmt = get_assmt(trace)
        @test get_args(trace) == (2, x_init_new, alpha, beta)
        @test assmt[1 => :x] == x1
        @test assmt[2 => :x] == x2
        expected_score = (logpdf(normal, x1, x_init_new * alpha + beta, std)
            + logpdf(normal, x2, x1 * alpha + beta, std))
        @test isapprox(get_score(trace), expected_score)
        expected_weight = (logpdf(normal, x1, x_init_new * alpha + beta, std)
            - logpdf(normal, x1, x_init * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        retval = get_retval(trace)
        @test length(retval) == 2
        @test retval[1] == x1
        @test retval[2] == x2
        @test isa(retdiff, NoRetDiff)
        @test isnodiff(retdiff)

        # init_changed=false, params_changed=true, change nothing
        trace = get_initial_trace()
        alpha_new = 0.5
        (trace, weight, retdiff) = free_update((2, x_init, alpha_new, beta),
            UnfoldCustomArgDiff(false, true), trace, EmptyAddressSet())
        assmt = get_assmt(trace)
        @test get_args(trace) == (2, x_init, alpha_new, beta)
        @test assmt[1 => :x] == x1
        @test assmt[2 => :x] == x2
        expected_score = (logpdf(normal, x1, x_init * alpha_new + beta, std)
            + logpdf(normal, x2, x1 * alpha_new + beta, std))
        @test isapprox(get_score(trace), expected_score)
        expected_weight = (logpdf(normal, x1, x_init * alpha_new + beta, std)
            - logpdf(normal, x1, x_init * alpha + beta, std)
            + logpdf(normal, x2, x1 * alpha_new + beta, std)
            - logpdf(normal, x2, x1 * alpha + beta, std))
        @test isapprox(weight, expected_weight)
        retval = get_retval(trace)
        @test length(retval) == 2
        @test retval[1] == x1
        @test retval[2] == x2
        @test isa(retdiff, NoRetDiff)
        @test isnodiff(retdiff)
    end


    @testset "extend" begin
        x_init = 0.1
        alpha = 0.2
        beta = 0.3
        x1 = 1.1
        x2 = 1.2

        function get_initial_trace()
            constraints = DynamicAssignment()
            constraints[1 => :x] = x1
            constraints[2 => :x] = x2
            (trace, _) = initialize(foo, (2, x_init, alpha, beta), constraints)
            trace
        end

        # increasing length from 2 to 4; constrain 4 and let 3 be generated from prior
        # also change alpha
        trace = get_initial_trace()
        x4_new = 1.3
        alpha_new = 0.5
        constraints = DynamicAssignment()
        constraints[4 => :x] = x4_new
        (trace, weight, retdiff) = extend((4, x_init, alpha_new, beta),
            unknownargdiff, trace, constraints)
        assmt = get_assmt(trace)
        @test get_args(trace) == (4, x_init, alpha_new, beta)
        @test assmt[1 => :x] == x1
        @test assmt[2 => :x] == x2
        x3_new = assmt[3 => :x]
        @test assmt[4 => :x] == x4_new
        expected_score = (logpdf(normal, x1, x_init * alpha_new + beta, std)
            + logpdf(normal, x2, x1 * alpha_new + beta, std)
            + logpdf(normal, x3_new, x2 * alpha_new + beta, std)
            + logpdf(normal, x4_new, x3_new * alpha_new + beta, std))
        @test isapprox(get_score(trace), expected_score)
        expected_weight = (logpdf(normal, x1, x_init * alpha_new + beta, std)
            - logpdf(normal, x1, x_init * alpha + beta, std)
            + logpdf(normal, x2, x1 * alpha_new + beta, std)
            - logpdf(normal, x2, x1 * alpha + beta, std)
            + logpdf(normal, x4_new, x3_new * alpha_new + beta, std))
        @test isapprox(weight, expected_weight)
        retval = get_retval(trace)
        @test length(retval) == 4
        @test retval[1] == x1
        @test retval[2] == x2
        @test retval[3] == x3_new
        @test retval[4] == x4_new
        @test isa(retdiff, VectorCustomRetDiff)
        @test !haskey(retdiff.retained_retdiffs, 1) # no diff
        @test !haskey(retdiff.retained_retdiffs, 2) # no diff
        @test !haskey(retdiff.retained_retdiffs, 3) # new, not retained
        @test !haskey(retdiff.retained_retdiffs, 4) # new, not retained
        @test !isnodiff(retdiff)
    end

    @testset "backprop_params" begin
        x_init = 0.1
        alpha = 0.2
        beta = 0.3
        x1 = 1.1
        x2 = 1.2

        constraints = DynamicAssignment()
        constraints[1 => :x] = x1
        constraints[2 => :x] = x2
        (trace, _) = initialize(foo, (2, x_init, alpha, beta), constraints)

        zero_param_grad!(kernel, :std)
        input_grads = backprop_params(trace, nothing)
        @test input_grads[1] == nothing # length
        @test input_grads[2] == nothing # inital state
        #@test isapprox(input_grads[3], expected_xs_grad) # alpha
        #@test isapprox(input_grads[4], expected_ys_grad) # beta
        expected_std_grad = (logpdf_grad(normal, x1, x_init * alpha + beta, std)[3]
            + logpdf_grad(normal, x2, x1 * alpha + beta, std)[3])
        @test isapprox(get_param_grad(kernel, :std), expected_std_grad)
    end
end
