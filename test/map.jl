@testset "map combinator" begin

    @gen function foo(x::Float64, y::Float64)
        @param std::Float64
        return @addr(normal(x + y, std), :z)
    end

    set_param!(foo, :std, 1.)

    @gen function bar()
        xs = [1.0, 2.0]
        ys = [3.0, 4.0]
        return @addr(Map(foo)(xs, ys), :map)
    end

    # test simulate
    assignment = get_assignment(simulate(bar, ()))
    @test has_leaf_node(assignment, :map => 1 => :z)
    @test has_leaf_node(assignment, :map => 2 => :z)

    # test generate
    z1, z2 = 1.1, 2.2
    constraints = DynamicAssignment()
    constraints[:map => 1 => :z] = z1
    constraints[:map => 2 => :z] = z2
    (trace, weight) = generate(bar, (), constraints)
    assignment = get_assignment(trace)
    @test assignment[:map => 1 => :z] == z1
    @test assignment[:map => 2 => :z] == z2
    @test isapprox(weight, logpdf(normal, z1, 4., 1.) + logpdf(normal, z2, 6., 1.))

    # test assess
    trace = assess(bar, (), constraints)
    assignment = get_assignment(trace)
    @test assignment[:map => 1 => :z] == z1
    @test assignment[:map => 2 => :z] == z2
    @test isapprox(weight, logpdf(normal, z1, 4., 1.) + logpdf(normal, z2, 6., 1.))

    # test update
    z2_new = 3.3
    constraints = DynamicAssignment()
    constraints[:map => 2 => :z] = z2_new
    (trace, weight, discard, retchange) = update(bar, (), nothing, trace, constraints)
    assignment = get_assignment(trace)
    @test assignment[:map => 1 => :z] == z1
    @test assignment[:map => 2 => :z] == z2_new
    @test discard[:map => 2 => :z] == z2
    @test isapprox(weight, logpdf(normal, z2_new, 6., 1.) - logpdf(normal, z2, 6., 1.))

    # test backprop_trace
    # TODO
end
