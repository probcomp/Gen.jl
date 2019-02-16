using Gen: generate_generative_function

@testset "static IR" begin

    #@gen function bar()
        #@trace(normal(0, 1), :a)
    #end
    
    builder = StaticIRBuilder()
    zero = add_constant_node!(builder, 0.)
    one = add_constant_node!(builder, 1.)
    add_addr_node!(builder, normal, inputs=[zero, one], addr=:a)
    ir = build_ir(builder)
    bar = eval(generate_generative_function(ir, :bar))

    #@gen function baz()
        #@trace(normal(0, 1), :b)
    #end

    builder = StaticIRBuilder()
    zero = add_constant_node!(builder, 0.)
    one = add_constant_node!(builder, 1.)
    add_addr_node!(builder, normal, inputs=[zero, one], addr=:b)
    ir = build_ir(builder)
    baz = eval(generate_generative_function(ir, :baz))

    #@gen function foo()
        #@trace(normal(0, 1), :x)
        #@trace(bar(), :u)
        #@trace(normal(0, 1), :y)
        #@trace(baz(), :v)
    #end

    builder = StaticIRBuilder()
    zero = add_constant_node!(builder, 0.)
    one = add_constant_node!(builder, 1.)
    add_addr_node!(builder, normal, inputs=[zero, one], addr=:x)
    add_addr_node!(builder, bar, inputs=[], addr=:u)
    add_addr_node!(builder, normal, inputs=[zero, one], addr=:y)
    add_addr_node!(builder, baz, inputs=[], addr=:v)
    ir = build_ir(builder)
    foo = eval(generate_generative_function(ir, :foo))

    #@gen function const_fn()
        #return 1
    #end

    builder = StaticIRBuilder()
    one = add_constant_node!(builder, 2)
    set_return_node!(builder, one)
    ir = build_ir(builder)
    const_fn = eval(generate_generative_function(ir, :const_fn))

    Gen.load_generated_functions()

@testset "Julia call" begin

    @test const_fn() == 2

end

@testset "simulate" begin

    trace = simulate(foo, ())
    x = trace[:x]
    a = trace[:u => :a]
    y = trace[:y]
    b = trace[:v => :b]

    score = (
        logpdf(normal, x, 0, 1) +
        logpdf(normal, a, 0, 1) +
        logpdf(normal, y, 0, 1) +
        logpdf(normal, b, 0, 1))

    @test isapprox(score, get_score(trace))
end

@testset "generate" begin

    y_constraint = 1.123
    b_constraint = -2.1
    constraints = choicemap()
    constraints[:y] = y_constraint
    constraints[:v => :b] = b_constraint
    (trace, weight) = generate(foo, (), constraints)
    x = trace[:x]
    a = trace[:u => :a]
    y = trace[:y]
    b = trace[:v => :b]

    @test isapprox(y, y_constraint)
    @test isapprox(b, b_constraint)

    score = (
        logpdf(normal, x, 0, 1) +
        logpdf(normal, a, 0, 1) +
        logpdf(normal, y, 0, 1) +
        logpdf(normal, b, 0, 1))

    @test isapprox(score, get_score(trace))

    prior_score = (
        logpdf(normal, x, 0, 1) +
        logpdf(normal, a, 0, 1))

    expected_weight = score - prior_score

    @test isapprox(weight, expected_weight)
end

@testset "project" begin

    y = 1.123
    b = -2.1
    choices = choicemap()
    choices[:y] = y
    choices[:v => :b] = b
    (trace, weight) = generate(foo, (), choices)
    x = get_choices(trace)[:x]
    a = get_choices(trace)[:u => :a]
    selection = select(:y, :u => :a)
    weight = project(trace, selection)

    expected_weight = logpdf(normal, y, 0., 1.) + logpdf(normal, a, 0., 1)
    @test isapprox(weight, expected_weight)
end

@testset "update" begin

    # get a trace
    constraints = choicemap()
    (trace,) = generate(foo, (), constraints)
    x_prev = get_choices(trace)[:x]
    a_prev = get_choices(trace)[:u => :a]
    y_prev = get_choices(trace)[:y]
    b_prev = get_choices(trace)[:v => :b]

    # force change to two of the variables
    y_new = 1.123
    b_new = -2.1
    constraints = choicemap()
    constraints[:y] = y_new
    constraints[:v => :b] = b_new
    (new_trace, weight, retdiff, discard) = update(trace,
        (), (), constraints)

    # test discard
    @test get_value(discard, :y) == y_prev
    @test get_value(discard, :v => :b) == b_prev
    @test length(collect(get_values_shallow(discard))) == 1
    @test length(collect(get_submaps_shallow(discard))) == 1

    # test new trace
    new_choices = get_choices(new_trace)
    @test get_value(new_choices, :y) == y_new
    @test get_value(new_choices, :v => :b) == b_new
    @test length(collect(get_values_shallow(new_choices))) == 2
    @test length(collect(get_submaps_shallow(new_choices))) == 2

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
    @test isapprox(prev_score, get_score(trace))
    @test isapprox(new_score, get_score(new_trace))
    @test isapprox(expected_weight, weight)

    # test retdiff
    @test retdiff === NoChange()
end

@testset "regenerate" begin

    Random.seed!(1)

    # get a trace
    constraints = choicemap()
    (trace,) = generate(foo, (), constraints)
    x_prev = get_choices(trace)[:x]
    a_prev = get_choices(trace)[:u => :a]
    y_prev = get_choices(trace)[:y]
    b_prev = get_choices(trace)[:v => :b]

    # resample :y and :v => :b
    selection = select(:y, :v => :b)
    (new_trace, weight, retdiff) = regenerate(trace,
        (), (), selection)

    # test new trace
    new_choices = get_choices(new_trace)
    @test get_value(new_choices, :x) == x_prev
    @test get_value(new_choices, :u => :a) == a_prev
    y_new = get_value(new_choices, :y)
    b_new = get_value(new_choices, :v => :b)
    @test y_new != y_prev
    @test b_new != b_prev
    @test length(collect(get_values_shallow(new_choices))) == 2
    @test length(collect(get_submaps_shallow(new_choices))) == 2

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
    @test isapprox(prev_score, get_score(trace))
    @test isapprox(new_score, get_score(new_trace))
    @test isapprox(0., weight)

    # test retdiff
    @test retdiff === NoChange()
end

@testset "extend" begin

    # get a trace
    constraints = choicemap()
    (trace,) = generate(foo, (), constraints)
    x_prev = get_choices(trace)[:x]
    a_prev = get_choices(trace)[:u => :a]
    y_prev = get_choices(trace)[:y]
    b_prev = get_choices(trace)[:v => :b]

    # don't do anything.. TODO write a better test
    constraints = choicemap()
    (new_trace, weight, retdiff) = extend(trace,
        (), (), constraints)

    # test new trace
    new_choices = get_choices(new_trace)
    @test get_value(new_choices, :x) == x_prev
    @test get_value(new_choices, :u => :a) == a_prev
    @test get_value(new_choices, :y) == y_prev
    @test get_value(new_choices, :v => :b) == b_prev
    @test length(collect(get_values_shallow(new_choices))) == 2
    @test length(collect(get_submaps_shallow(new_choices))) == 2

    # test score and weight
    score = (
        logpdf(normal, x_prev, 0, 1) +
        logpdf(normal, a_prev, 0, 1) +
        logpdf(normal, y_prev, 0, 1) +
        logpdf(normal, b_prev, 0, 1))
    @test isapprox(score, get_score(new_trace))
    @test isapprox(0., weight)

    # test retdiff
    @test retdiff === NoChange()
end

@testset "backprop" begin

    # bar
    builder = StaticIRBuilder()
    mu_z = add_argument_node!(builder, name=:mu_z, typ=:Float64, compute_grad=true)
    one = add_constant_node!(builder, 1.)
    z = add_addr_node!(builder, normal, inputs=[mu_z, one], addr=:z, name=:z)
    retval = add_julia_node!(builder, (z, mu_z) -> z + mu_z, inputs=[z, mu_z], name=:retval)
    set_return_node!(builder, retval)
    ir = build_ir(builder)
    bar = eval(generate_generative_function(ir, :bar))
    
    # foo
    builder = StaticIRBuilder()
    mu_a = add_argument_node!(builder, name=:mu_a, typ=:Float64, compute_grad=true)
    one = add_constant_node!(builder, 1.)
    a = add_addr_node!(builder, normal, inputs=[mu_a, one], addr=:a, name=:a)
    b = add_addr_node!(builder, normal, inputs=[a, one], addr=:b, name=:b)
    bar_val = add_addr_node!(builder, bar, inputs=[a], addr=:bar, name=:bar_val)
    c = add_julia_node!(builder, (a, b, bar) -> (a * b * bar), inputs=[a, b, bar_val], name=:c)
    retval = add_addr_node!(builder, normal, inputs=[c, one], addr=:out, name=:out)
    set_return_node!(builder, retval)
    ir = build_ir(builder)
    foo = eval(generate_generative_function(ir, :foo))

    Gen.load_generated_functions()

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
    constraints = choicemap()
    constraints[:a] = a
    constraints[:b] = b
    constraints[:out] = out
    constraints[:bar => :z] = z
    (trace, _) = generate(foo, (mu_a,), constraints)

    # compute gradients with choice_gradients
    selection = select(:bar => :z, :a, :out)
    selection = StaticAddressSet(selection)
    retval_grad = 2.
    ((mu_a_grad,), value_trie, gradient_trie) = choice_gradients(trace, selection, retval_grad)

    # check input gradient
    @test isapprox(mu_a_grad, finite_diff(f, (mu_a, a, b, z, out), 1, dx))

    # check value trie
    @test get_value(value_trie, :a) == a
    @test get_value(value_trie, :out) == out
    @test get_value(value_trie, :bar => :z) == z
    @test !has_value(value_trie, :b) # was not selected
    @test length(get_submaps_shallow(value_trie)) == 1
    @test length(get_values_shallow(value_trie)) == 2

    # check gradient trie
    @test length(get_submaps_shallow(gradient_trie)) == 1
    @test length(get_values_shallow(gradient_trie)) == 2
    @test !has_value(gradient_trie, :b) # was not selected
    @test isapprox(get_value(gradient_trie, :a), finite_diff(f, (mu_a, a, b, z, out), 2, dx))
    @test isapprox(get_value(gradient_trie, :out), finite_diff(f, (mu_a, a, b, z, out), 5, dx))
    @test isapprox(get_value(gradient_trie, :bar => :z), finite_diff(f, (mu_a, a, b, z, out), 4, dx))

    # compute gradients with accumulate_param_gradients!
    retval_grad = 2.
    (mu_a_grad,) = accumulate_param_gradients!(trace, retval_grad)

    # check input gradient
    @test isapprox(mu_a_grad, finite_diff(f, (mu_a, a, b, z, out), 1, dx))
end

end # @testset "static IR"
