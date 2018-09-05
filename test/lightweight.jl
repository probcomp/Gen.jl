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
    constraints = DynamicChoiceTrie()
    constraints[:branch] = true
    (trace,) = generate(foo, (), constraints)
    x = get_choices(trace)[:x]
    a = get_choices(trace)[:u => :a]

    # force to follow the second branch
    y = 1.123
    b = -2.1
    constraints = DynamicChoiceTrie()
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
    new_choices = get_choices(new_trace)
    @test get_leaf_node(new_choices, :branch) == false
    @test get_leaf_node(new_choices, :y) == y
    @test get_leaf_node(new_choices, :v => :b) == b
    @test length(collect(get_leaf_nodes(new_choices))) == 2
    @test length(collect(get_internal_nodes(new_choices))) == 1

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
    end

    # get a trace which follows the first branch
    constraints = DynamicChoiceTrie()
    constraints[:branch] = true
    (trace,) = generate(foo, (), constraints)
    x = get_choices(trace)[:x]
    a = get_choices(trace)[:u => :a]

    # force to follow the second branch
    y = 1.123
    b = -2.1
    constraints = DynamicChoiceTrie()
    constraints[:branch] = false
    (new_trace, weight, discard, retchange) = fix_update(
        foo, (), nothing, trace, constraints)

    # test discard
    @test get_leaf_node(discard, :branch) == true
    @test length(collect(get_leaf_nodes(discard))) == 1
    @test length(collect(get_internal_nodes(discard))) == 0

    # test new trace
    new_choices = get_choices(new_trace)
    @test get_leaf_node(new_choices, :branch) == false
    y = get_leaf_node(new_choices, :y)
    b = get_leaf_node(new_choices, :v => :b)
    @test length(collect(get_leaf_nodes(new_choices))) == 2
    @test length(collect(get_internal_nodes(new_choices))) == 1

    # test score and weight
    expected_new_score = (
        logpdf(bernoulli, false, 0.4) +
        logpdf(normal, y, 0, 1) +
        logpdf(normal, b, 0, 1))
    expected_weight = logpdf(bernoulli, false, 0.4) - logpdf(bernoulli, true, 0.4)
    @test isapprox(expected_new_score, get_call_record(new_trace).score)
    @test isapprox(expected_weight, weight)

    # test retchange (should be nothing by default)
    @test retchange === nothing
end

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
    constraints = DynamicChoiceTrie()
    constraints[:branch] = true
    (trace,) = generate(foo, (mu,), constraints)
    x = get_choices(trace)[:x]
    a = get_choices(trace)[:u => :a]

    # resimulate branch
    selection = DynamicAddressSet()
    push_leaf_node!(selection, :branch)

    # try 10 times, so we are likely to get both a stay and a switch
    for i=1:10
        prev_choices = get_choices(trace)

        # change the argument so that the weights can be nonzer
        prev_mu = mu
        mu = rand()
        (trace, weight, retchange) = regenerate(foo, (mu,), nothing, trace, selection)
        choices = get_choices(trace)

        # test score
        if choices[:branch]
            expected_score = logpdf(normal, choices[:x], mu, 1) + logpdf(normal, choices[:u => :a], mu, 1) + logpdf(bernoulli, true, 0.4)
        else
            expected_score = logpdf(normal, choices[:y], mu, 1) + logpdf(normal, choices[:v => :b], mu, 1) + logpdf(bernoulli, false, 0.4)
        end
        @test isapprox(expected_score, get_call_record(trace).score)

        # test values
        if choices[:branch]
            @test has_leaf_node(choices, :x)
            @test has_internal_node(choices, :u)
        else
            @test has_leaf_node(choices, :y)
            @test has_internal_node(choices, :v)
        end
        @test length(collect(get_leaf_nodes(choices))) == 2
        @test length(collect(get_internal_nodes(choices))) == 1

        # test weight
        if choices[:branch] == prev_choices[:branch]
            if choices[:branch]
                expected_weight = logpdf(normal, choices[:x], mu, 1) + logpdf(normal, choices[:u => :a], mu, 1)
                expected_weight -= logpdf(normal, choices[:x], prev_mu, 1) + logpdf(normal, choices[:u => :a], prev_mu, 1)
            else
                expected_weight = logpdf(normal, choices[:y], mu, 1) + logpdf(normal, choices[:v => :b], mu, 1)                
                expected_weight -= logpdf(normal, choices[:y], prev_mu, 1) + logpdf(normal, choices[:v => :b], prev_mu, 1)
            end
        else 
            expected_weight = 0.
        end
        @test isapprox(expected_weight, weight)

        # test retchange (should be nothing by default)
        @test retchange === nothing
    end

end

