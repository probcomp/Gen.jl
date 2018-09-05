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

    # resimulate branch
    selection = DynamicAddressSet()
    push_leaf_node!(selection, :branch)
    # TODO

end

