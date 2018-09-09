@testset "basic block force update" begin

    eval(:(@gen function bar()
        if @addr(bernoulli(0.4), :branch)
            @addr(normal(0, 1), :x)
        else
            @addr(normal(0, 1), :y)
        end
    end))
    
    eval(:(@compiled @gen function foo2()
        @addr(bar(), :a)
        @addr(normal(0, 1), :b)
        @addr(normal(0, 1), :c)
    end))

    Gen.load_generated_functions()

    # get a trace which follows the first branch
    constraints = DynamicChoiceTrie()
    constraints[:a => :branch] = true
    (trace,) = generate(foo2, (), constraints)
    x = get_choices(trace)[:a => :x]
    b = get_choices(trace)[:b]
    c = get_choices(trace)[:c]

    # force to follow the second branch
    b_new = 0.123
    constraints = DynamicChoiceTrie()
    constraints[:a => :branch] = false
    constraints[:b] = b_new
    constraints[:a => :y] = 2.3
    (new_trace, weight, discard, retchange) = update(
        foo2, (), nothing, trace, constraints)

    # test discard
    @test get_leaf_node(discard, :a => :branch) == true
    @test get_leaf_node(discard, :b) == b
    @test get_leaf_node(discard, :a => :x) == x
    @test length(collect(get_leaf_nodes(discard))) == 1
    @test length(collect(get_internal_nodes(discard))) == 1

    # test new trace
    new_choices = get_choices(new_trace)
    @test get_leaf_node(new_choices, :a => :branch) == false
    @test get_leaf_node(new_choices, :b) == b_new
    @test length(collect(get_leaf_nodes(new_choices))) == 2
    @test length(collect(get_internal_nodes(new_choices))) == 1
    y = new_choices[:a => :y]

    # test score and weight
    prev_score = (
        logpdf(bernoulli, true, 0.4) +
        logpdf(normal, x, 0, 1) +
        logpdf(normal, b, 0, 1) +
        logpdf(normal, c, 0, 1))
    expected_new_score = (
        logpdf(bernoulli, false, 0.4) +
        logpdf(normal, y, 0, 1) +
        logpdf(normal, b_new, 0, 1) +
        logpdf(normal, c, 0, 1))
    expected_weight = expected_new_score - prev_score
    @test isapprox(expected_new_score, get_call_record(new_trace).score)
    @test isapprox(expected_weight, weight)

    # test retchange (should be nothing by default)
    @test retchange === nothing
end
