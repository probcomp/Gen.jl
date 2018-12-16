@testset "SGD training" begin

    Random.seed!(1)

    # x y   z

    # 0 0   0
    # 0 1   1
    # 1 0   1
    # 1 1   0

    # x y   p(x, y | z=0)
    # 0 0   0.5
    # 0 1   0
    # 1 0   0
    # 1 1   0.5

    # x y   p(x, y | z=1)
    # 0 0   0
    # 0 1   0.5
    # 1 0   0.5
    # 1 1   0


    @gen function teacher()
        x = @addr(bernoulli(0.5), :x)
        y = @addr(bernoulli(0.5), :y)
        z::Bool = xor(x, y)
        @addr(bernoulli(z ? 1.0 : 0.0), :z)
    end

    sigmoid(val) = 1.0 / (1.0 + exp(-val))

    @gen function batch_student(zs::Vector{Bool})
        @param theta1::Float64
        @param theta2::Float64
        @param theta3::Float64
        @param theta4::Float64
        @param theta5::Float64
        for (i, z) in enumerate(zs)
            x = @addr(bernoulli(sigmoid(theta1)), i => :x)
            if z && x
                prob_y = sigmoid(theta2) # should be near 0
            elseif z && !x
                prob_y = sigmoid(theta3) # should be near 1
            elseif !z && x
                prob_y = sigmoid(theta4) # should be near 1
            elseif !z && !x
                prob_y = sigmoid(theta5) # should be near 0
            end
            @addr(bernoulli(prob_y), i => :y)
        end
    end

    # theta1 = 0.0
    # theta2 -> inf (prob_y -> 1)
    # theta3 -> -inf (prob_y -> 0)
    
    input_extractor = (samples::Vector) -> (Bool[s[:z] for s in samples],)

    function constraint_extractor(samples::Vector)
        constraints = DynamicAssignment()
        for (i, s) in enumerate(samples)
            constraints[i => :x] = s[:x]
            constraints[i => :y] = s[:y]
        end
        constraints
    end

    function minibatch_callback(batch, minibatch, avg_score, verbose)
        for name in [:theta1, :theta2, :theta3, :theta4, :theta5]
            grad = get_param_grad(batch_student, name)
            value = get_param(batch_student, name)
            set_param!(batch_student, name, value + grad * 0.01)
            zero_param_grad!(batch_student, name)
        end
    end

    batch_callback = (batch, verbose) -> nothing

    init_param!(batch_student, :theta1, 0.)
    init_param!(batch_student, :theta2, 0.)
    init_param!(batch_student, :theta3, 0.)
    init_param!(batch_student, :theta4, 0.)
    init_param!(batch_student, :theta5, 0.)

    # check gradients using finite differences on a simulated batch
    assignments = Vector{Any}(undef, 100)
    for i=1:100
        (assignments[i], _) = propose(teacher, ())
    end
    input = input_extractor(assignments)
    constraints = constraint_extractor(assignments)
    (student_trace, _) = initialize(batch_student, input, constraints)
    backprop_params(student_trace, nothing)
    for name in [:theta1, :theta2, :theta3, :theta4, :theta5]
        actual = get_param_grad(batch_student, name)
        dx = 1e-6
        value = get_param(batch_student, name)
        set_param!(batch_student, name, value + dx)
        (lpdf_pos, _) = assess(batch_student, input, constraints)
        set_param!(batch_student, name, value - dx)
        (lpdf_neg, _) = assess(batch_student, input, constraints)
        set_param!(batch_student, name, value)
        expected = (lpdf_pos - lpdf_neg) / (2 * dx)
        @test isapprox(actual, expected, atol=1e-4)
    end

    conf = SGDTrainConf(2000, 50, 1, 50,
        input_extractor, constraint_extractor,
        minibatch_callback, batch_callback)
    sgd_train_batch(teacher, (), batch_student, conf, false)

    # p(x | z=0) = p(x | z=1) = 0.5
    @test isapprox(get_param(batch_student, :theta1), 0., atol=0.2)

    # y | z, x = xor(x, z)
    @test get_param(batch_student, :theta2) < -5
    @test get_param(batch_student, :theta3) > 5
    @test get_param(batch_student, :theta4) > 5
    @test get_param(batch_student, :theta5) < -5
end
