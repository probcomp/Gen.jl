@testset "backprop" begin

    #@gen (static) function bar(mu_z::Float64)
        #z = @trace(normal(mu_z, 1), :z)
        #return z + mu_z
    #end

    # bar
    builder = StaticIRBuilder()
    mu_z = add_argument_node!(builder, name=:mu_z, typ=:Float64, compute_grad=true)
    one = add_constant_node!(builder, 1.)
    z = add_addr_node!(builder, normal, inputs=[mu_z, one], addr=:z, name=:z)
    retval = add_julia_node!(builder, (z, mu_z) -> z + mu_z, inputs=[z, mu_z], name=:retval)
    set_return_node!(builder, retval)
    ir = build_ir(builder)
    bar = eval(generate_generative_function(ir, :bar, track_diffs=false, cache_julia_nodes=false))

    #@gen (static) function foo(mu_a::Float64)
        #param theta::Float64
        #a = @trace(normal(mu_a, 1), :a)
        #b = @trace(normal(a, 1), :b)
        #bar = @trace(bar(a), :bar)
        #c = a * b * bar * theta
        #out = @trace(normal(c, 1), :out)
        #return out
    #end

    # foo
    builder = StaticIRBuilder()
    mu_a = add_argument_node!(builder, name=:mu_a, typ=:Float64, compute_grad=true)
    theta = add_trainable_param_node!(builder, :theta, typ=QuoteNode(Float64))
    one = add_constant_node!(builder, 1.)
    a = add_addr_node!(builder, normal, inputs=[mu_a, one], addr=:a, name=:a)
    b = add_addr_node!(builder, normal, inputs=[a, one], addr=:b, name=:b)
    bar_val = add_addr_node!(builder, bar, inputs=[a], addr=:bar, name=:bar_val)
    c = add_julia_node!(builder, (a, b, bar, theta) -> (a * b * bar * theta),
            inputs=[a, b, bar_val, theta], name=:c)
    retval = add_addr_node!(builder, normal, inputs=[c, one], addr=:out, name=:out)
    set_return_node!(builder, retval)
    ir = build_ir(builder)
    foo = eval(generate_generative_function(ir, :foo, track_diffs=false, cache_julia_nodes=false))

    Gen.load_generated_functions()
   
    # test get_parameters
    store_to_ids = Gen.get_parameters(foo, Gen.default_parameter_context)
    @test length(store_to_ids) == 1
    @test length(store_to_ids[Gen.default_julia_parameter_store]) == 1
    @test (foo, :theta) in store_to_ids[Gen.default_julia_parameter_store]

    function f(mu_a, theta, a, b, z, out)
        lpdf = 0.
        mu_z = a
        lpdf += logpdf(normal, z, mu_z, 1)
        lpdf += logpdf(normal, a, mu_a, 1)
        lpdf += logpdf(normal, b, a, 1)
        c = a * b * (z + mu_z) * theta
        lpdf += logpdf(normal, out, c, 1)
        return lpdf + 2 * out
    end

    mu_a = 1.
    theta = -0.5
    a = 2.
    b = 3.
    z = 4.
    out = 5.

    # initialize the trainable parameter
    init_parameter!((foo, :theta), theta)

    # get the initial trace
    constraints = choicemap()
    constraints[:a] = a
    constraints[:b] = b
    constraints[:out] = out
    constraints[:bar => :z] = z
    (trace, _) = generate(foo, (mu_a,), constraints)

    # compute gradients with choice_gradients
    selection = select(:bar => :z, :a, :out)
    selection = StaticSelection(selection)
    retval_grad = 2.
    ((mu_a_grad,), value_trie, gradient_trie) = choice_gradients(trace, selection, retval_grad)

    # check input gradient
    @test isapprox(mu_a_grad, finite_diff(f, (mu_a, theta, a, b, z, out), 1, dx))

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
    @test isapprox(get_value(gradient_trie, :a), finite_diff(f, (mu_a, theta, a, b, z, out), 3, dx))
    @test isapprox(get_value(gradient_trie, :out), finite_diff(f, (mu_a, theta, a, b, z, out), 6, dx))
    @test isapprox(get_value(gradient_trie, :bar => :z), finite_diff(f, (mu_a, theta, a, b, z, out), 5, dx))

    # compute gradients with accumulate_param_gradients!
    retval_grad = 2.
    (mu_a_grad,) = accumulate_param_gradients!(trace, retval_grad)

    # check input gradient
    @test isapprox(mu_a_grad, finite_diff(f, (mu_a, theta, a, b, z, out), 1, dx))

    # check trainable parameter gradient
    @test isapprox(
        get_gradient((foo, :theta)),
        finite_diff(f, (mu_a, theta, a, b, z, out), 2, dx))

end
