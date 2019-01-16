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

    @gen function student(z::Bool)
        @param theta1::Float64
        @param theta2::Float64
        @param theta3::Float64
        @param theta4::Float64
        @param theta5::Float64
        x = @addr(bernoulli(sigmoid(theta1)), :x)
        if z && x
            prob_y = sigmoid(theta2) # should be near 0
        elseif z && !x
            prob_y = sigmoid(theta3) # should be near 1
        elseif !z && x
            prob_y = sigmoid(theta4) # should be near 1
        elseif !z && !x
            prob_y = sigmoid(theta5) # should be near 0
        end
        @addr(bernoulli(prob_y), :y)
    end

    function data_generator()
        (assmt, _, retval) = propose(teacher, ())
        local inputs
        local constraints
        inputs = (assmt[:z],)
        constraints = DynamicAssignment()
        constraints[:x] = assmt[:x]
        constraints[:y] = assmt[:y]
        (inputs, constraints)
    end

    # theta1 = 0.0
    # theta2 -> inf (prob_y -> 1)
    # theta3 -> -inf (prob_y -> 0)
    
    init_param!(student, :theta1, 0.)
    init_param!(student, :theta2, 0.)
    init_param!(student, :theta3, 0.)
    init_param!(student, :theta4, 0.)
    init_param!(student, :theta5, 0.)

    # check gradients using finite differences on a simulated batch
    minibatch_size = 100
    inputs = Vector{Any}(undef, minibatch_size)
    constraints = Vector{Any}(undef, minibatch_size)
    for i=1:minibatch_size
        (inputs[i], constraints[i]) = data_generator()
        (student_trace, _) = initialize(student, inputs[i], constraints[i])
        backprop_params(student_trace, nothing)
    end
    for name in [:theta1, :theta2, :theta3, :theta4, :theta5]
        actual = get_param_grad(student, name)
        dx = 1e-6
        value = get_param(student, name)

        # evaluate total log density at value + dx
        set_param!(student, name, value + dx)
        lpdf_pos = 0.
        for i=1:minibatch_size
            (incr, _) = assess(student, inputs[i], constraints[i])
            lpdf_pos += incr
        end

        # evaluate total log density at value - dx
        set_param!(student, name, value - dx)
        lpdf_neg = 0.
        for i=1:minibatch_size
            (incr, _) = assess(student, inputs[i], constraints[i])
            lpdf_neg += incr
        end
    
        expected = (lpdf_pos - lpdf_neg) / (2 * dx)
        @test isapprox(actual, expected, atol=1e-4)

        set_param!(student, name, value)
    end

    # use stochastic gradient descent
    opt = Optimizer(GradientDescentConf(0.01, 1000000), student)
    train!(student, data_generator, opt, 2000, 50, 1, 50;
        verbose=false)

    # p(x | z=0) = p(x | z=1) = 0.5
    @test isapprox(get_param(student, :theta1), 0., atol=0.2)

    # y | z, x = xor(x, z)
    @test get_param(student, :theta2) < -5
    @test get_param(student, :theta3) > 5
    @test get_param(student, :theta4) > 5
    @test get_param(student, :theta5) < -5
end
