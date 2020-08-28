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
        x = @trace(bernoulli(0.5), :x)
        y = @trace(bernoulli(0.5), :y)
        z::Bool = xor(x, y)
        @trace(bernoulli(z ? 1.0 : 0.0), :z)
    end

    sigmoid(val) = 1.0 / (1.0 + exp(-val))

    @gen function student(z::Bool)
        @param theta1::Float64
        @param theta2::Float64
        @param theta3::Float64
        @param theta4::Float64
        @param theta5::Float64
        x = @trace(bernoulli(sigmoid(theta1)), :x)
        if z && x
            prob_y = sigmoid(theta2) # should be near 0
        elseif z && !x
            prob_y = sigmoid(theta3) # should be near 1
        elseif !z && x
            prob_y = sigmoid(theta4) # should be near 1
        elseif !z && !x
            prob_y = sigmoid(theta5) # should be near 0
        end
        @trace(bernoulli(prob_y), :y)
    end

    function data_generator()
        (choices, _, retval) = propose(teacher, ())
        local inputs
        local constraints
        inputs = (choices[:z],)
        constraints = choicemap()
        constraints[:x] = choices[:x]
        constraints[:y] = choices[:y]
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
        (student_trace, _) = generate(student, inputs[i], constraints[i])
        accumulate_param_gradients!(student_trace, nothing)
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
    update = ParamUpdate(GradientDescent(0.01, 1000000), student)
    train!(student, data_generator, update,
        num_epoch=2000, epoch_size=50, num_minibatch=1, minibatch_size=50,
        verbose=false)

    # p(x | z=0) = p(x | z=1) = 0.5
    @test isapprox(get_param(student, :theta1), 0., atol=0.2)

    # y | z, x = xor(x, z)
    @test get_param(student, :theta2) < -5
    @test get_param(student, :theta3) > 5
    @test get_param(student, :theta4) > 5
    @test get_param(student, :theta5) < -5
end


@testset "lecture!" begin

    Random.seed!(1)

    # simple p
    @gen function p()
        z = @trace(normal(0., 1.), :z)
        x = @trace(normal(z + 2., 0.3), :x)
        return x
    end

    # simple q
    @gen function q(x::Float64)
        @param theta::Float64
        @param log_std::Float64
        z = @trace(normal(x + theta, exp(log_std)), :z)
        return z
    end

    # train simple q using lecture! to compute gradients
    init_param!(q, :theta, 0.)
    init_param!(q, :log_std, 0.)
    update = ParamUpdate(FixedStepGradientDescent(1e-4), q)
    score = Inf
    for iter=1:100
        score = sum([lecture!(p, (), q, tr -> (tr[:x],)) for _=1:1000]) / 1000
        apply!(update)
    end
    score = sum([lecture!(p, (), q, tr -> (tr[:x],)) for _=1:10000]) / 10000
    @test isapprox(score, -0.21, atol=5e-2)

    # simple q, batched
    @gen function q_batched(xs::Vector{Float64})
        @param theta::Float64
        @param log_std::Float64
        means = xs .+ theta
        for i=1:length(xs)
            @trace(normal(means[i], exp(log_std)), i => :z)
        end
    end

    # train simple q using lecture_batched! to compute gradients
    init_param!(q_batched, :theta, 0.)
    init_param!(q_batched, :log_std, 0.)
    update = ParamUpdate(FixedStepGradientDescent(0.001), q_batched)
    score = Inf
    for iter=1:100
        score = lecture_batched!(p, (), q_batched, trs -> (map(tr -> tr[:x], trs),), 1000)
        apply!(update)
    end
    score = sum([lecture!(p, (), q, tr -> (tr[:x],)) for _=1:10000]) / 10000
    @test isapprox(score, -0.21, atol=5e-2)
end
