@testset "lightweight gen function" begin

##########
# update #
##########

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
    (trace,) = generate(foo, (), constraints)
    x = get_assignment(trace)[:x]
    a = get_assignment(trace)[:u => :a]

    # force to follow the second branch
    y = 1.123
    b = -2.1
    constraints = DynamicAssignment()
    constraints[:branch] = false
    constraints[:y] = y
    constraints[:v => :b] = b
    (new_trace, weight, discard, retchange) = update(
        foo, (), nothing, trace, constraints)

    # test discard
    @test get_leaf_node(discard, :branch) == true
    @test get_leaf_node(discard, :x) == x
    @test get_leaf_node(discard, :u => :a) == a
    @test length(collect(get_leaf_nodes(discard))) == 2
    @test length(collect(get_internal_nodes(discard))) == 1

    # test new trace
    new_assignment = get_assignment(new_trace)
    @test get_leaf_node(new_assignment, :branch) == false
    @test get_leaf_node(new_assignment, :y) == y
    @test get_leaf_node(new_assignment, :v => :b) == b
    @test length(collect(get_leaf_nodes(new_assignment))) == 2
    @test length(collect(get_internal_nodes(new_assignment))) == 1

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
    @test isapprox(expected_new_score, get_call_record(new_trace).score)
    @test isapprox(expected_weight, weight)

    # test retchange (should be nothing by default)
    @test retchange === nothing
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
    (trace,) = generate(foo, (), constraints)
    x = get_assignment(trace)[:x]
    a = get_assignment(trace)[:u => :a]
    z = get_assignment(trace)[:z]

    # force to follow the second branch, and change z
    y = 1.123
    b = -2.1
    z_new = 0.4
    constraints = DynamicAssignment()
    constraints[:branch] = false
    constraints[:z] = z_new
    (new_trace, weight, discard, retchange) = fix_update(
        foo, (), nothing, trace, constraints)

    # test discard
    @test get_leaf_node(discard, :branch) == true
    @test get_leaf_node(discard, :z) == z
    @test length(collect(get_leaf_nodes(discard))) == 2
    @test length(collect(get_internal_nodes(discard))) == 0

    # test new trace
    new_assignment = get_assignment(new_trace)
    @test get_leaf_node(new_assignment, :branch) == false
    @test get_leaf_node(new_assignment, :z) == z_new
    y = get_leaf_node(new_assignment, :y)
    b = get_leaf_node(new_assignment, :v => :b)
    @test length(collect(get_leaf_nodes(new_assignment))) == 3
    @test length(collect(get_internal_nodes(new_assignment))) == 1

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
    @test isapprox(expected_new_score, get_call_record(new_trace).score)
    @test isapprox(expected_weight, weight)

    # test retchange (should be nothing by default)
    @test retchange === nothing
end


###############
# free_update #
###############

@testset "regenerate" begin

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
    (trace,) = generate(foo, (mu,), constraints)
    x = get_assignment(trace)[:x]
    a = get_assignment(trace)[:u => :a]

    # resimulate branch
    selection = DynamicAddressSet()
    push_leaf_node!(selection, :branch)

    # try 10 times, so we are likely to get both a stay and a switch
    for i=1:10
        prev_assignment = get_assignment(trace)

        # change the argument so that the weights can be nonzer
        prev_mu = mu
        mu = rand()
        (trace, weight, retchange) = regenerate(foo, (mu,), nothing, trace, selection)
        assignment = get_assignment(trace)

        # test score
        if assignment[:branch]
            expected_score = logpdf(normal, assignment[:x], mu, 1) + logpdf(normal, assignment[:u => :a], mu, 1) + logpdf(bernoulli, true, 0.4)
        else
            expected_score = logpdf(normal, assignment[:y], mu, 1) + logpdf(normal, assignment[:v => :b], mu, 1) + logpdf(bernoulli, false, 0.4)
        end
        @test isapprox(expected_score, get_call_record(trace).score)

        # test values
        if assignment[:branch]
            @test has_leaf_node(assignment, :x)
            @test has_internal_node(assignment, :u)
        else
            @test has_leaf_node(assignment, :y)
            @test has_internal_node(assignment, :v)
        end
        @test length(collect(get_leaf_nodes(assignment))) == 2
        @test length(collect(get_internal_nodes(assignment))) == 1

        # test weight
        if assignment[:branch] == prev_assignment[:branch]
            if assignment[:branch]
                expected_weight = logpdf(normal, assignment[:x], mu, 1) + logpdf(normal, assignment[:u => :a], mu, 1)
                expected_weight -= logpdf(normal, assignment[:x], prev_mu, 1) + logpdf(normal, assignment[:u => :a], prev_mu, 1)
            else
                expected_weight = logpdf(normal, assignment[:y], mu, 1) + logpdf(normal, assignment[:v => :b], mu, 1)                
                expected_weight -= logpdf(normal, assignment[:y], prev_mu, 1) + logpdf(normal, assignment[:v => :b], prev_mu, 1)
            end
        else 
            expected_weight = 0.
        end
        @test isapprox(expected_weight, weight)

        # test retchange (should be nothing by default)
        @test retchange === nothing
    end

end

end
