using Gen: AddressVisitor, all_visited, visit!, get_visited

@testset "Dynamic DSL" begin

###################
# address visitor #
###################

# TODO also test get_unvisited

@testset "address visitor" begin
    assmt = DynamicAssignment()
    assmt[:x] = 1
    assmt[:y => :z] = 2

    visitor = AddressVisitor()
    visit!(visitor, :x)
    visit!(visitor, :y => :z)
    @test all_visited(get_visited(visitor), assmt)

    visitor = AddressVisitor()
    visit!(visitor, :x)
    @test !all_visited(get_visited(visitor), assmt)

    visitor = AddressVisitor()
    visit!(visitor, :y => :z)
    @test !all_visited(get_visited(visitor), assmt)

    visitor = AddressVisitor()
    visit!(visitor, :x)
    visit!(visitor, :y)
    @test all_visited(get_visited(visitor), assmt)
end


################
# force update #
################

@testset "force update" begin

    @gen function bar()
        @addr(normal(0, 1), :a)
    end

    @gen function baz()
        @addr(normal(0, 1), :b)
    end

    @gen function foo()
        if @addr(bernoulli(0.4), :branch)
            @addr(normal(0, 1), :x)
            @addr(bar(), :u)
        else
            @addr(normal(0, 1), :y)
            @addr(baz(), :v)
        end
    end

    # get a trace which follows the first branch
    constraints = DynamicAssignment()
    constraints[:branch] = true
    (trace,) = initialize(foo, (), constraints)
    x = get_assmt(trace)[:x]
    a = get_assmt(trace)[:u => :a]

    # force to follow the second branch
    y = 1.123
    b = -2.1
    constraints = DynamicAssignment()
    constraints[:branch] = false
    constraints[:y] = y
    constraints[:v => :b] = b
    (new_trace, weight, discard, retdiff) = force_update(
        (), unknownargdiff, trace, constraints)

    # test discard
    @test get_value(discard, :branch) == true
    @test get_value(discard, :x) == x
    @test get_value(discard, :u => :a) == a
    @test length(collect(get_values_shallow(discard))) == 2
    @test length(collect(get_subassmts_shallow(discard))) == 1

    # test new trace
    new_assignment = get_assmt(new_trace)
    @test get_value(new_assignment, :branch) == false
    @test get_value(new_assignment, :y) == y
    @test get_value(new_assignment, :v => :b) == b
    @test length(collect(get_values_shallow(new_assignment))) == 2
    @test length(collect(get_subassmts_shallow(new_assignment))) == 1

    # test score and weight
    prev_score = (
        logpdf(bernoulli, true, 0.4) +
        logpdf(normal, x, 0, 1) +
        logpdf(normal, a, 0, 1))
    expected_new_score = (
        logpdf(bernoulli, false, 0.4) +
        logpdf(normal, y, 0, 1) +
        logpdf(normal, b, 0, 1))
    expected_weight = expected_new_score - prev_score
    @test isapprox(expected_new_score, get_score(new_trace))
    @test isapprox(expected_weight, weight)

    # test retdiff
    @test retdiff === DefaultRetDiff()
end


##############
# fix_update #
##############

@testset "fix update" begin

    @gen function bar()
        @addr(normal(0, 1), :a)
    end

    @gen function baz()
        @addr(normal(0, 1), :b)
    end

    @gen function foo()
        if @addr(bernoulli(0.4), :branch)
            @addr(normal(0, 1), :x)
            @addr(bar(), :u)
        else
            @addr(normal(0, 1), :y)
            @addr(baz(), :v)
        end
        @addr(normal(0, 1), :z)
    end

    # get a trace which follows the first branch
    constraints = DynamicAssignment()
    constraints[:branch] = true
    (trace,) = initialize(foo, (), constraints)
    x = get_assmt(trace)[:x]
    a = get_assmt(trace)[:u => :a]
    z = get_assmt(trace)[:z]

    # force to follow the second branch, and change z
    y = 1.123
    b = -2.1
    z_new = 0.4
    constraints = DynamicAssignment()
    constraints[:branch] = false
    constraints[:z] = z_new
    (new_trace, weight, discard, retdiff) = fix_update(
        (), unknownargdiff, trace, constraints)

    # test discard
    @test get_value(discard, :branch) == true
    @test get_value(discard, :z) == z
    @test length(collect(get_values_shallow(discard))) == 2
    @test length(collect(get_subassmts_shallow(discard))) == 0

    # test new trace
    new_assignment = get_assmt(new_trace)
    @test get_value(new_assignment, :branch) == false
    @test get_value(new_assignment, :z) == z_new
    y = get_value(new_assignment, :y)
    b = get_value(new_assignment, :v => :b)
    @test length(collect(get_values_shallow(new_assignment))) == 3
    @test length(collect(get_subassmts_shallow(new_assignment))) == 1

    # test score and weight
    expected_new_score = (
        logpdf(bernoulli, false, 0.4) +
        logpdf(normal, y, 0, 1) +
        logpdf(normal, b, 0, 1) +
        logpdf(normal, z_new, 0, 1))
    expected_weight = (
        logpdf(bernoulli, false, 0.4)
        - logpdf(bernoulli, true, 0.4)
        + logpdf(normal, z_new, 0, 1)
        - logpdf(normal, z, 0, 1))
    @test isapprox(expected_new_score, get_score(new_trace))
    @test isapprox(expected_weight, weight)

    # test retdiff
    @test retdiff === DefaultRetDiff()
end


###############
# free_update #
###############

@testset "free_update" begin

    @gen function bar(mu)
        @addr(normal(mu, 1), :a)
    end

    @gen function baz(mu)
        @addr(normal(mu, 1), :b)
    end

    @gen function foo(mu)
        if @addr(bernoulli(0.4), :branch)
            @addr(normal(mu, 1), :x)
            @addr(bar(mu), :u)
        else
            @addr(normal(mu, 1), :y)
            @addr(baz(mu), :v)
        end
    end

    # get a trace which follows the first branch
    mu = 0.123
    constraints = DynamicAssignment()
    constraints[:branch] = true
    (trace,) = initialize(foo, (mu,), constraints)
    x = get_assmt(trace)[:x]
    a = get_assmt(trace)[:u => :a]

    # resimulate branch
    selection = select(:branch)

    # try 10 times, so we are likely to get both a stay and a switch
    for i=1:10
        prev_assignment = get_assmt(trace)

        # change the argument so that the weights can be nonzer
        prev_mu = mu
        mu = rand()
        (trace, weight, retdiff) = free_update(
            (mu,), unknownargdiff, trace, selection)
        assignment = get_assmt(trace)

        # test score
        if assignment[:branch]
            expected_score = (
                logpdf(normal, assignment[:x], mu, 1)
                + logpdf(normal, assignment[:u => :a], mu, 1)
                + logpdf(bernoulli, true, 0.4))
        else
            expected_score = (
                logpdf(normal, assignment[:y], mu, 1)
                + logpdf(normal, assignment[:v => :b], mu, 1)
                + logpdf(bernoulli, false, 0.4))
        end
        @test isapprox(expected_score, get_score(trace))

        # test values
        if assignment[:branch]
            @test has_value(assignment, :x)
            @test !isempty(get_subassmt(assignment, :u))
        else
            @test has_value(assignment, :y)
            @test !isempty(get_subassmt(assignment, :v))
        end
        @test length(collect(get_values_shallow(assignment))) == 2
        @test length(collect(get_subassmts_shallow(assignment))) == 1

        # test weight
        if assignment[:branch] == prev_assignment[:branch]
            if assignment[:branch]
                expected_weight = (
                    logpdf(normal, assignment[:x], mu, 1)
                    + logpdf(normal, assignment[:u => :a], mu, 1))
                expected_weight -= (
                    logpdf(normal, assignment[:x], prev_mu, 1)
                    + logpdf(normal, assignment[:u => :a], prev_mu, 1))
            else
                expected_weight = (
                    logpdf(normal, assignment[:y], mu, 1)
                    + logpdf(normal, assignment[:v => :b], mu, 1))
                expected_weight -= (
                    logpdf(normal, assignment[:y], prev_mu, 1)
                    + logpdf(normal, assignment[:v => :b], prev_mu, 1))
            end
        else 
            expected_weight = 0.
        end
        @test isapprox(expected_weight, weight)

        # test retchange (should be nothing by default)
        @test retdiff === DefaultRetDiff()
    end

end

##################
# backprop_trace #
##################

@testset "backprop trace" begin

    @gen function bar(@grad(mu_z::Float64))
        @param theta1::Float64
        z = @addr(normal(mu_z + theta1, 1), :z)
        return z + mu_z
    end

    @gen function foo(@grad(mu_a::Float64))
        @param theta2::Float64
        a = @addr(normal(mu_a, 1), :a)
        b = @addr(normal(a, 1), :b)
        c = a * b * @addr(bar(a), :bar)
        return @addr(normal(c, 1), :out) + (theta2 * 3)
    end

    init_param!(bar, :theta1, 0.)
    init_param!(foo, :theta2, 0.)

    function f(mu_a, a, b, z, out)
        lpdf = 0.
        mu_z = a
        lpdf += logpdf(normal, z, mu_z, 1)
        lpdf += logpdf(normal, a, mu_a, 1)
        lpdf += logpdf(normal, b, a, 1)
        c = a * b * (z + mu_z)
        lpdf += logpdf(normal, out, c, 1)
        return lpdf + 2 * out
    end

    mu_a = 1.
    a = 2.
    b = 3.
    z = 4.
    out = 5.

    # get the initial trace
    constraints = DynamicAssignment()
    constraints[:a] = a
    constraints[:b] = b
    constraints[:out] = out
    constraints[:bar => :z] = z
    (trace, _) = initialize(foo, (mu_a,), constraints)

    # compute gradients using backprop_trace
    selection = select(:bar => :z, :a, :out)
    retval_grad = 2.
    ((mu_a_grad,), value_assmt, gradient_assmt) = backprop_trace(
        trace, selection, retval_grad)

    # check input gradient
    @test isapprox(mu_a_grad, finite_diff(f, (mu_a, a, b, z, out), 1, dx))

    # check value trie
    @test get_value(value_assmt, :a) == a
    @test get_value(value_assmt, :out) == out
    @test get_value(value_assmt, :bar => :z) == z
    @test !has_value(value_assmt, :b) # was not selected
    @test length(collect(get_subassmts_shallow(value_assmt))) == 1
    @test length(collect(get_values_shallow(value_assmt))) == 2

    # check gradient trie
    @test length(collect(get_subassmts_shallow(gradient_assmt))) == 1
    @test length(collect(get_values_shallow(gradient_assmt))) == 2
    @test !has_value(gradient_assmt, :b) # was not selected
    @test isapprox(get_value(gradient_assmt, :bar => :z),
        finite_diff(f, (mu_a, a, b, z, out), 4, dx))
    @test isapprox(get_value(gradient_assmt, :out),
        finite_diff(f, (mu_a, a, b, z, out), 5, dx))

    # compute gradients using backprop_params
    selection = select(:bar => :z, :a, :out)
    retval_grad = 2.
    (mu_a_grad,) = backprop_params(trace, retval_grad)

    # check input gradient
    @test isapprox(mu_a_grad, finite_diff(f, (mu_a, a, b, z, out), 1, dx))

    # check parameter gradient
    theta1_grad = get_param_grad(bar, :theta1)
    theta2_grad = get_param_grad(foo, :theta2)
    @test isapprox(theta1_grad, logpdf_grad(normal, z, a, 1)[2])
    @test isapprox(theta2_grad, 3 * 2)

end

@testset "backprop params with splice" begin

    @gen function baz()
        @param theta::Float64
        return theta
    end

    init_param!(baz, :theta, 0.)

    @gen function foo()
        return @splice(baz())
    end

    (trace, _) = initialize(foo, ())
    retval_grad = 2.
    backprop_params(trace, retval_grad)
    @test isapprox(get_param_grad(baz, :theta), retval_grad)
end

end
