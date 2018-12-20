@testset "call_at combinator" begin

    @gen function foo(@grad(x::Float64))
        return x + @addr(normal(x, 1), :y)
    end

    at = call_at(foo, Int)

    @testset "assess" begin
        y = 1.234
        assmt = DynamicAssignment()
        assmt[3 => :y] = y
        (weight, value) = assess(at, (0.4, 3), assmt)
        @test isapprox(weight, logpdf(normal, y, 0.4, 1))
        @test isapprox(value, 0.4 + y)
    end

    @testset "propose" begin
        (assmt, weight, value) = propose(at, (0.4, 3))
        y = assmt[3 => :y]
        @test isapprox(weight, logpdf(normal, y, 0.4, 1))
        @test length(collect(get_values_shallow(assmt))) == 0
        @test length(collect(get_subassmts_shallow(assmt))) == 1
    end

    @testset "initialize" begin

        # without constraint
        (trace, weight) = initialize(at, (0.4, 3), EmptyAssignment())
        assmt = get_assmt(trace)
        @test weight == 0.
        y = assmt[3 => :y]
        @test get_retval(trace) == 0.4 + y
        @test length(collect(get_values_shallow(assmt))) == 0
        @test length(collect(get_subassmts_shallow(assmt))) == 1

        # with constraints
        y = 1.234
        constraints = DynamicAssignment()
        constraints[3 => :y] = y
        (trace, weight) = initialize(at, (0.4, 3), constraints)
        assmt = get_assmt(trace)
        @test assmt[3 => :y] == y
        @test get_retval(trace) == 0.4 + y
        @test isapprox(weight, logpdf(normal, y, 0.4, 1.))
        @test length(collect(get_values_shallow(assmt))) == 0
        @test length(collect(get_subassmts_shallow(assmt))) == 1
    end

    function get_trace()
        y = 1.234
        constraints = DynamicAssignment()
        constraints[3 => :y] = y
        (trace, _) = initialize(at, (0.4, 3), constraints)
        (trace, y)
    end

    @testset "project" begin
        (trace, y) = get_trace()
        @test isapprox(project(trace, EmptyAddressSet()), 0.)
        selection = select(3 => :y)
        @test isapprox(project(trace, selection), logpdf(normal, y, 0.4, 1))
    end

    @testset "force update" begin
        (trace, y) = get_trace()

        # change kernel_args, same key, no constraint
        (new_trace, weight, discard, retdiff) = force_update((0.2, 3), unknownargdiff,
            trace, EmptyAssignment())
        assmt = get_assmt(new_trace)
        @test assmt[3 => :y] == y
        @test length(collect(get_values_shallow(assmt))) == 0
        @test length(collect(get_subassmts_shallow(assmt))) == 1
        @test isapprox(weight, logpdf(normal, y, 0.2, 1) - logpdf(normal, y, 0.4, 1))
        @test get_retval(new_trace) == 0.2 + y
        @test isempty(discard)
        @test isapprox(get_score(new_trace), logpdf(normal, y, 0.2, 1))

        # change kernel_args, same key, with constraint
        constraints = DynamicAssignment()
        y_new = 2.345
        constraints[3 => :y] = y_new
        (new_trace, weight, discard, retdiff) = force_update((0.2, 3), unknownargdiff,
            trace, constraints)
        assmt = get_assmt(new_trace)
        @test assmt[3 => :y] == y_new
        @test length(collect(get_values_shallow(assmt))) == 0
        @test length(collect(get_subassmts_shallow(assmt))) == 1
        @test isapprox(weight, logpdf(normal, y_new, 0.2, 1) - logpdf(normal, y, 0.4, 1))
        @test get_retval(new_trace) == 0.2 + y_new
        @test discard[3 => :y] == y
        @test length(collect(get_values_shallow(discard))) == 0
        @test length(collect(get_subassmts_shallow(discard))) == 1
        @test isapprox(get_score(new_trace), logpdf(normal, y_new, 0.2, 1))

        # change kernel_args, different key, with constraint
        y_new = 2.345
        constraints = DynamicAssignment()
        constraints[4 => :y] = y_new
        (new_trace, weight, discard, retdiff) = force_update((0.2, 4), unknownargdiff,
            trace, constraints)
        assmt = get_assmt(new_trace)
        @test assmt[4 => :y] == y_new
        @test length(collect(get_values_shallow(assmt))) == 0
        @test length(collect(get_subassmts_shallow(assmt))) == 1
        @test isapprox(weight, logpdf(normal, y_new, 0.2, 1) - logpdf(normal, y, 0.4, 1))
        @test get_retval(new_trace) == 0.2 + y_new
        @test discard[3 => :y] == y
        @test length(collect(get_values_shallow(discard))) == 0
        @test length(collect(get_subassmts_shallow(discard))) == 1
        @test isapprox(get_score(new_trace), logpdf(normal, y_new, 0.2, 1))
    end

    @testset "fix update" begin
        (trace, y) = get_trace()

        # change kernel_args, same key, no constraint
        (new_trace, weight, discard, retdiff) = fix_update((0.2, 3), unknownargdiff,
            trace, EmptyAssignment())
        assmt = get_assmt(new_trace)
        @test assmt[3 => :y] == y
        @test length(collect(get_values_shallow(assmt))) == 0
        @test length(collect(get_subassmts_shallow(assmt))) == 1
        @test isapprox(weight, logpdf(normal, y, 0.2, 1) - logpdf(normal, y, 0.4, 1))
        @test get_retval(new_trace) == 0.2 + y
        @test isempty(discard)
        @test isapprox(get_score(new_trace), logpdf(normal, y, 0.2, 1))

        # change kernel_args, same key, with constraint
        constraints = DynamicAssignment()
        y_new = 2.345
        constraints[3 => :y] = y_new
        (new_trace, weight, discard, retdiff) = fix_update((0.2, 3), unknownargdiff,
            trace, constraints)
        assmt = get_assmt(new_trace)
        @test assmt[3 => :y] == y_new
        @test length(collect(get_values_shallow(assmt))) == 0
        @test length(collect(get_subassmts_shallow(assmt))) == 1
        @test isapprox(weight, logpdf(normal, y_new, 0.2, 1) - logpdf(normal, y, 0.4, 1))
        @test get_retval(new_trace) == 0.2 + y_new
        @test discard[3 => :y] == y
        @test length(collect(get_values_shallow(discard))) == 0
        @test length(collect(get_subassmts_shallow(discard))) == 1
        @test isapprox(get_score(new_trace), logpdf(normal, y_new, 0.2, 1))

        # change kernel_args, different key, no constraint
        (new_trace, weight, discard, retdiff) = fix_update((0.2, 4), unknownargdiff,
            trace, EmptyAssignment())
        assmt = get_assmt(new_trace)
        y_new = assmt[4 => :y]
        @test length(collect(get_values_shallow(assmt))) == 0
        @test length(collect(get_subassmts_shallow(assmt))) == 1
        @test weight == 0.
        @test get_retval(new_trace) == 0.2 + y_new
        @test isempty(discard)
        @test isapprox(get_score(new_trace), logpdf(normal, y_new, 0.2, 1))
    end

    @testset "free update" begin
        (trace, y) = get_trace()

        # change kernel_args, same key, not selected
        (new_trace, weight, retdiff) = free_update((0.2, 3), unknownargdiff,
            trace, EmptyAddressSet())
        assmt = get_assmt(new_trace)
        @test assmt[3 => :y] == y
        @test length(collect(get_values_shallow(assmt))) == 0
        @test length(collect(get_subassmts_shallow(assmt))) == 1
        @test isapprox(weight, logpdf(normal, y, 0.2, 1) - logpdf(normal, y, 0.4, 1))
        @test get_retval(new_trace) == 0.2 + y
        @test isapprox(get_score(new_trace), logpdf(normal, y, 0.2, 1))

        # change kernel_args, same key, selected
        selection = select(3 => :y)
        (new_trace, weight, retdiff) = free_update((0.2, 3), unknownargdiff,
            trace, selection)
        assmt = get_assmt(new_trace)
        y_new = assmt[3 => :y]
        @test length(collect(get_values_shallow(assmt))) == 0
        @test length(collect(get_subassmts_shallow(assmt))) == 1
        @test weight == 0.
        @test get_retval(new_trace) == 0.2 + y_new
        @test isapprox(get_score(new_trace), logpdf(normal, y_new, 0.2, 1))

        # change kernel_args, different key, not selected
        (new_trace, weight, retdiff) = free_update((0.2, 4), unknownargdiff,
            trace, EmptyAddressSet())
        assmt = get_assmt(new_trace)
        y_new = assmt[4 => :y]
        @test length(collect(get_values_shallow(assmt))) == 0
        @test length(collect(get_subassmts_shallow(assmt))) == 1
        @test weight == 0.
        @test get_retval(new_trace) == 0.2 + y_new
        @test isapprox(get_score(new_trace), logpdf(normal, y_new, 0.2, 1))
    end

    @testset "extend" begin
        (trace, y) = get_trace()

        # change kernel_args, same key, no constraint (the only valid input)
        (new_trace, weight, retdiff) = extend((0.2, 3), unknownargdiff,
            trace, EmptyAssignment())
        assmt = get_assmt(new_trace)
        @test assmt[3 => :y] == y
        @test length(collect(get_values_shallow(assmt))) == 0
        @test length(collect(get_subassmts_shallow(assmt))) == 1
        @test isapprox(weight, logpdf(normal, y, 0.2, 1) - logpdf(normal, y, 0.4, 1))
        @test get_retval(new_trace) == 0.2 + y
    end

    @testset "backprop_trace" begin
        (trace, y) = get_trace()

        # not selected
        (input_grads, value_assmt, gradient_assmt) = backprop_trace(
            trace, EmptyAddressSet(), nothing)
        @test isempty(value_assmt)
        @test isempty(gradient_assmt)
        @test length(input_grads) == 2
        @test isapprox(input_grads[1], logpdf_grad(normal, y, 0.4, 1.0)[2])
        @test input_grads[2] == nothing # the key has no gradient

        # selected without retval_grad
        selection = select(3 => :y)
        (input_grads, value_assmt, gradient_assmt) = backprop_trace(
            trace, selection, nothing)
        @test value_assmt[3 => :y] == y
        @test isapprox(gradient_assmt[3 => :y], logpdf_grad(normal, y, 0.4, 1.0)[1])
        @test length(collect(get_values_shallow(gradient_assmt))) == 0
        @test length(collect(get_subassmts_shallow(gradient_assmt))) == 1
        @test length(collect(get_values_shallow(value_assmt))) == 0
        @test length(collect(get_subassmts_shallow(value_assmt))) == 1
        @test length(input_grads) == 2
        @test isapprox(input_grads[1], logpdf_grad(normal, y, 0.4, 1.0)[2])
        @test input_grads[2] == nothing # the key has no gradient

        # selected with retval_grad
        retval_grad = 1.234
        selection = select(3 => :y)
        (input_grads, value_assmt, gradient_assmt) = backprop_trace(
            trace, selection, retval_grad)
        @test value_assmt[3 => :y] == y
        @test isapprox(gradient_assmt[3 => :y], logpdf_grad(normal, y, 0.4, 1.0)[1] + retval_grad)
        @test length(collect(get_values_shallow(gradient_assmt))) == 0
        @test length(collect(get_subassmts_shallow(gradient_assmt))) == 1
        @test length(collect(get_values_shallow(value_assmt))) == 0
        @test length(collect(get_subassmts_shallow(value_assmt))) == 1
        @test length(input_grads) == 2
        @test isapprox(input_grads[1], logpdf_grad(normal, y, 0.4, 1.0)[2] + retval_grad)
        @test input_grads[2] == nothing # the key has no gradient
    end

    @testset "backprop_params" begin
        (trace, y) = get_trace()
        retval_grad = 1.234
        input_grads = backprop_params(trace, retval_grad)
        @test length(input_grads) == 2
        @test isapprox(input_grads[1], logpdf_grad(normal, y, 0.4, 1)[2] + retval_grad)
        @test input_grads[2] == nothing # the key has no gradient
    end
end
