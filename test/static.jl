using Gen: generate_generative_function

@testset "static IR" begin

    #@gen function bar()
        #@addr(normal(0, 1), :a)
    #end
    
    builder = StaticIRBuilder()
    zero = add_constant_node!(builder, 0.)
    one = add_constant_node!(builder, 1.)
    add_random_choice_node!(builder, normal, [zero, one], :a, gensym(), Float64)
    ir = build_ir(builder)
    eval(generate_generative_function(ir, :bar))

    #@gen function baz()
        #@addr(normal(0, 1), :b)
    #end

    builder = StaticIRBuilder()
    zero = add_constant_node!(builder, 0.)
    one = add_constant_node!(builder, 1.)
    add_random_choice_node!(builder, normal, [zero, one], :b, gensym(), Float64)
    ir = build_ir(builder)
    eval(generate_generative_function(ir, :baz))

    #@gen function foo()
        #@addr(normal(0, 1), :x)
        #@addr(bar(), :u)
        #@addr(normal(0, 1), :y)
        #@addr(baz(), :v)
    #end

    builder = StaticIRBuilder()
    zero = add_constant_node!(builder, 0.)
    one = add_constant_node!(builder, 1.)
    add_random_choice_node!(builder, normal, [zero, one], :x, gensym(), Float64)
    bar_argdiff = add_constant_node!(builder, unknownargdiff)
    add_gen_fn_call_node!(builder, bar, [], :u, bar_argdiff, gensym(), Nothing)
    add_random_choice_node!(builder, normal, [zero, one], :y, gensym(), Float64)
    baz_argdiff = add_constant_node!(builder, unknownargdiff)
    add_gen_fn_call_node!(builder, baz, [], :v, baz_argdiff, gensym(), Nothing)
    ir = build_ir(builder)
    eval(generate_generative_function(ir, :foo))

    Gen.load_generated_functions()


@testset "generate" begin

    y_constraint = 1.123
    b_constraint = -2.1
    constraints = DynamicAssignment()
    constraints[:y] = y_constraint
    constraints[:v => :b] = b_constraint
    #constraints = StaticAssignment(constraints)
    (trace, weight) = generate(foo, (), constraints)
    x = get_assignment(trace)[:x]
    a = get_assignment(trace)[:u => :a]
    y = get_assignment(trace)[:y]
    b = get_assignment(trace)[:v => :b]

    @test isapprox(y, y_constraint)
    @test isapprox(b, b_constraint)

    score = (
        logpdf(normal, x, 0, 1) +
        logpdf(normal, a, 0, 1) +
        logpdf(normal, y, 0, 1) +
        logpdf(normal, b, 0, 1))

    @test isapprox(score, get_call_record(trace).score)

    prior_score = (
        logpdf(normal, x, 0, 1) +
        logpdf(normal, a, 0, 1))

    expected_weight = score - prior_score

    @test isapprox(weight, expected_weight)
end

@testset "force update" begin

    # get a trace
    constraints = DynamicAssignment()
    (trace,) = generate(foo, (), constraints)
    x_prev = get_assignment(trace)[:x]
    a_prev = get_assignment(trace)[:u => :a]
    y_prev = get_assignment(trace)[:y]
    b_prev = get_assignment(trace)[:v => :b]

    # force change to two of the variables
    y_new = 1.123
    b_new = -2.1
    constraints = DynamicAssignment()
    constraints[:y] = y_new
    constraints[:v => :b] = b_new
    (new_trace, weight, discard, retdiff) = update(
        foo, (), unknownargdiff, trace, constraints)

    # test discard
    @test get_leaf_node(discard, :y) == y_prev
    @test get_leaf_node(discard, :v => :b) == b_prev
    @test length(collect(get_leaf_nodes(discard))) == 1
    @test length(collect(get_internal_nodes(discard))) == 1

    # test new trace
    new_assignment = get_assignment(new_trace)
    @test get_leaf_node(new_assignment, :y) == y_new
    @test get_leaf_node(new_assignment, :v => :b) == b_new
    @test length(collect(get_leaf_nodes(new_assignment))) == 2
    @test length(collect(get_internal_nodes(new_assignment))) == 2

    # test score and weight
    prev_score = (
        logpdf(normal, x_prev, 0, 1) +
        logpdf(normal, a_prev, 0, 1) +
        logpdf(normal, y_prev, 0, 1) +
        logpdf(normal, b_prev, 0, 1))
    new_score = (
        logpdf(normal, x_prev, 0, 1) +
        logpdf(normal, a_prev, 0, 1) +
        logpdf(normal, y_new, 0, 1) +
        logpdf(normal, b_new, 0, 1))
    expected_weight = new_score - prev_score
    @test isapprox(prev_score, get_call_record(trace).score)
    @test isapprox(new_score, get_call_record(new_trace).score)
    @test isapprox(expected_weight, weight)

    # test retdiff
    @test retdiff === DefaultRetDiff()
end

end # @testset "static IR"
