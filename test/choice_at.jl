@testset "choice_at combinator" begin

    at = choice_at(bernoulli, Int)

    @testset "assess" begin
        assmt = DynamicAssignment()
        assmt[3] = true
        (weight, value) = assess(at, (0.4, 3), assmt)
        @test isapprox(weight, log(0.4))
        @test value == true
    end

    @testset "propose" begin
        (assmt, weight, value) = propose(at, (0.4, 3))
        @test isapprox(weight, value ? log(0.4) : log(0.6))
        @test assmt[3] == value
        @test length(collect(get_values_shallow(assmt))) == 1
        @test length(collect(get_subassmts_shallow(assmt))) == 0
    end

    @testset "generate" begin

        # without assignment
        (trace, weight) = generate(at, (0.4, 3), EmptyAssignment())
        @test weight == 0.
        value = get_retval(trace)
        assmt = get_assmt(trace)
        @test assmt[3] == value
        @test length(collect(get_values_shallow(assmt))) == 1
        @test length(collect(get_subassmts_shallow(assmt))) == 0

        # with constraints
        constraints = DynamicAssignment()
        constraints[3] = true
        (trace, weight) = generate(at, (0.4, 3), constraints)
        value = get_retval(trace)
        @test value == true
        @test isapprox(weight, log(0.4))
        assmt = get_assmt(trace)
        @test assmt[3] == value
        @test length(collect(get_values_shallow(assmt))) == 1
        @test length(collect(get_subassmts_shallow(assmt))) == 0
    end

    function get_trace()
        constraints = DynamicAssignment()
        constraints[3] = true
        (trace, _) = generate(at, (0.4, 3), constraints)
        trace
    end

    @testset "project" begin
        trace = get_trace()
        @test isapprox(project(trace, EmptyAddressSet()), 0.)
        selection = select(3)
        @test isapprox(project(trace, selection), log(0.4))
    end

    @testset "update" begin
        trace = get_trace()

        # change kernel_args, same key, no constraint
        (new_trace, weight, retdiff, discard) = update(trace,
            (0.2, 3), unknownargdiff, EmptyAssignment())
        assmt = get_assmt(new_trace)
        @test assmt[3] == true
        @test length(collect(get_values_shallow(assmt))) == 1
        @test length(collect(get_subassmts_shallow(assmt))) == 0
        @test isapprox(weight, log(0.2) - log(0.4))
        @test get_retval(new_trace) == true
        @test isempty(discard)

        # change kernel_args, same key, with constraint
        constraints = DynamicAssignment()
        constraints[3] = false
        (new_trace, weight, retdiff, discard) = update(trace,
            (0.2, 3), unknownargdiff, constraints)
        assmt = get_assmt(new_trace)
        @test assmt[3] == false
        @test length(collect(get_values_shallow(assmt))) == 1
        @test length(collect(get_subassmts_shallow(assmt))) == 0
        @test isapprox(weight, log(1 - 0.2) - log(0.4))
        @test get_retval(new_trace) == false
        @test discard[3] == true
        @test length(collect(get_values_shallow(discard))) == 1
        @test length(collect(get_subassmts_shallow(discard))) == 0

        # change kernel_args, different key, with constraint
        constraints = DynamicAssignment()
        constraints[4] = false
        (new_trace, weight, retdiff, discard) = update(trace,
            (0.2, 4), unknownargdiff, constraints)
        assmt = get_assmt(new_trace)
        @test assmt[4] == false
        @test length(collect(get_values_shallow(assmt))) == 1
        @test length(collect(get_subassmts_shallow(assmt))) == 0
        @test isapprox(weight, log(1 - 0.2) - log(0.4))
        @test get_retval(new_trace) == false
        @test discard[3] == true
        @test length(collect(get_values_shallow(discard))) == 1
        @test length(collect(get_subassmts_shallow(discard))) == 0
    end

    @testset "regenerate" begin
        trace = get_trace()

        # change kernel_args, same key, not selected
        (new_trace, weight, retdiff) = regenerate(trace,
            (0.2, 3), unknownargdiff, EmptyAddressSet())
        assmt = get_assmt(new_trace)
        @test assmt[3] == true
        @test length(collect(get_values_shallow(assmt))) == 1
        @test length(collect(get_subassmts_shallow(assmt))) == 0
        @test isapprox(weight, log(0.2) - log(0.4))
        @test get_retval(new_trace) == true
        @test isapprox(get_score(new_trace), log(0.2))

        # change kernel_args, same key, selected
        selection = select(3)
        (new_trace, weight, retdiff) = regenerate(trace,
            (0.2, 3), unknownargdiff, selection)
        assmt = get_assmt(new_trace)
        value = assmt[3]
        @test length(collect(get_values_shallow(assmt))) == 1
        @test length(collect(get_subassmts_shallow(assmt))) == 0
        @test weight == 0.
        @test get_retval(new_trace) == value
        @test isapprox(get_score(new_trace), log(value ? 0.2 : 1 - 0.2))

        # change kernel_args, different key, not selected
        (new_trace, weight, retdiff) = regenerate(trace,
            (0.2, 4), unknownargdiff, EmptyAddressSet())
        assmt = get_assmt(new_trace)
        value = assmt[4]
        @test length(collect(get_values_shallow(assmt))) == 1
        @test length(collect(get_subassmts_shallow(assmt))) == 0
        @test weight == 0.
        @test get_retval(new_trace) == value
        @test isapprox(get_score(new_trace), log(value ? 0.2 : 1 - 0.2))
    end

    @testset "extend" begin
        trace = get_trace()

        # change kernel_args, same key, no constraint (the only valid input)
        (new_trace, weight, retdiff) = extend(trace,
            (0.2, 3), unknownargdiff, EmptyAssignment())
        assmt = get_assmt(new_trace)
        @test assmt[3] == true
        @test length(collect(get_values_shallow(assmt))) == 1
        @test length(collect(get_subassmts_shallow(assmt))) == 0
        @test isapprox(weight, log(0.2) - log(0.4))
        @test get_retval(new_trace) == true
    end

    @testset "choice_gradients" begin
        y = 1.2
        constraints = DynamicAssignment()
        set_value!(constraints, 3, y)
        (trace, _) = generate(choice_at(normal, Int), (0.0, 1.0, 3), constraints)

        # not selected
        (input_grads, value_assmt, gradient_assmt) = choice_gradients(
            trace, EmptyAddressSet(), 1.2)
        @test isempty(value_assmt)
        @test isempty(gradient_assmt)
        @test length(input_grads) == 3
        @test isapprox(input_grads[1], logpdf_grad(normal, y, 0.0, 1.0)[2])
        @test isapprox(input_grads[2], logpdf_grad(normal, y, 0.0, 1.0)[3])
        @test input_grads[3] == nothing # the key has no gradient

        # selected with retval_grad
        retval_grad = 1.234
        selection = select(3)
        (input_grads, value_assmt, gradient_assmt) = choice_gradients(
            trace, selection, retval_grad)
        @test value_assmt[3] == y
        @test isapprox(gradient_assmt[3], logpdf_grad(normal, y, 0.0, 1.0)[1] + retval_grad)
        @test length(collect(get_values_shallow(gradient_assmt))) == 1
        @test length(collect(get_subassmts_shallow(gradient_assmt))) == 0
        @test length(collect(get_values_shallow(value_assmt))) == 1
        @test length(collect(get_subassmts_shallow(value_assmt))) == 0
        @test length(input_grads) == 3
        @test isapprox(input_grads[1], logpdf_grad(normal, y, 0.0, 1.0)[2])
        @test isapprox(input_grads[2], logpdf_grad(normal, y, 0.0, 1.0)[3])
        @test input_grads[3] == nothing # the key has no gradient
    end

    @testset "accumulate_param_gradients!" begin
        trace = get_trace()
        input_grads = accumulate_param_gradients!(trace, nothing)
        @test length(input_grads) == 2
        @test isapprox(input_grads[1], logpdf_grad(bernoulli, true, 0.4)[2])
        @test input_grads[2] == nothing # the key has no gradient
    end

end
