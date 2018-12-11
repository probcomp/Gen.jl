using Gen: generate_generative_function

@testset "static IR" begin

    #@gen function bar()
        #@addr(normal(0, 1), :a)
    #end
    
    builder = StaticIRBuilder()
    zero = add_constant_node!(builder, 0.)
    one = add_constant_node!(builder, 1.)
    add_random_choice_node!(builder, normal, inputs=[zero, one], addr=:a, typ=Float64)
    ir = build_ir(builder)
    eval(generate_generative_function(ir, :bar))

    #@gen function baz()
        #@addr(normal(0, 1), :b)
    #end

    builder = StaticIRBuilder()
    zero = add_constant_node!(builder, 0.)
    one = add_constant_node!(builder, 1.)
    add_random_choice_node!(builder, normal, inputs=[zero, one], addr=:b, typ=Float64)
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
    add_random_choice_node!(builder, normal, inputs=[zero, one], addr=:x, typ=Float64)
    add_gen_fn_call_node!(builder, bar, inputs=[], addr=:u, typ=Nothing)
    add_random_choice_node!(builder, normal, inputs=[zero, one], addr=:y, typ=Float64)
    add_gen_fn_call_node!(builder, baz, inputs=[], addr=:v, typ=Nothing)
    ir = build_ir(builder)
    eval(generate_generative_function(ir, :foo))

    Gen.load_generated_functions()


@testset "initialize" begin

    y_constraint = 1.123
    b_constraint = -2.1
    constraints = DynamicAssignment()
    constraints[:y] = y_constraint
    constraints[:v => :b] = b_constraint
    (trace, weight) = initialize(foo, (), constraints)
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
    assmt = DynamicAssignment()
    assmt[:y] = y
    assmt[:v => :b] = b
    (trace, weight) = initialize(foo, (), assmt)
    x = get_assignment(trace)[:x]
    a = get_assignment(trace)[:u => :a]
    selection = DynamicAddressSet()
    push_leaf_node!(selection, :y)
    push_leaf_node!(selection, :u => :a)
    weight = project(trace, selection)

    expected_weight = logpdf(normal, y, 0., 1.) + logpdf(normal, a, 0., 1)
    @test isapprox(weight, expected_weight)
end

#@testset "force update" begin
#
    ## get a trace
    #constraints = DynamicAssignment()
    #(trace,) = initialize(foo, (), constraints)
    #x_prev = get_assignment(trace)[:x]
    #a_prev = get_assignment(trace)[:u => :a]
    #y_prev = get_assignment(trace)[:y]
    #b_prev = get_assignment(trace)[:v => :b]
#
    ## force change to two of the variables
    #y_new = 1.123
    #b_new = -2.1
    #constraints = DynamicAssignment()
    #constraints[:y] = y_new
    #constraints[:v => :b] = b_new
    #(new_trace, weight, discard, retdiff) = force_update(
        #foo, (), unknownargdiff, trace, constraints)
#
    ## test discard
    #@test get_value(discard, :y) == y_prev
    #@test get_value(discard, :v => :b) == b_prev
    #@test length(collect(get_values_shallow(discard))) == 1
    #@test length(collect(get_subassmts_shallow(discard))) == 1
#
    ## test new trace
    #new_assignment = get_assignment(new_trace)
    #@test get_value(new_assignment, :y) == y_new
    #@test get_value(new_assignment, :v => :b) == b_new
    #@test length(collect(get_values_shallow(new_assignment))) == 2
    #@test length(collect(get_subassmts_shallow(new_assignment))) == 2
#
    ## test score and weight
    #prev_score = (
        #logpdf(normal, x_prev, 0, 1) +
        #logpdf(normal, a_prev, 0, 1) +
        #logpdf(normal, y_prev, 0, 1) +
        #logpdf(normal, b_prev, 0, 1))
    #new_score = (
        #logpdf(normal, x_prev, 0, 1) +
        #logpdf(normal, a_prev, 0, 1) +
        #logpdf(normal, y_new, 0, 1) +
        #logpdf(normal, b_new, 0, 1))
    #expected_weight = new_score - prev_score
    #@test isapprox(prev_score, get_score(trace))
    #@test isapprox(new_score, get_score(new_trace))
    #@test isapprox(expected_weight, weight)
#
    ## test retdiff
    #@test retdiff === DefaultRetDiff()
#end
#
#@testset "backprop" begin
#
    ## bar
    #builder = StaticIRBuilder()
    #mu_z = add_argument_node!(builder, name=:mu_z, typ=Float64, compute_grad=true)
    #one = add_constant_node!(builder, 1.)
    #z = add_random_choice_node!(builder, normal, inputs=[mu_z, one], addr=:z, typ=Float64, name=:z)
    #retval = add_julia_node!(builder, (z, mu_z) -> z + mu_z, inputs=[z, mu_z], name=:retval)
    #set_return_node!(builder, retval)
    #ir = build_ir(builder)
    #eval(generate_generative_function(ir, :bar))
    #
    ## foo
    #builder = StaticIRBuilder()
    #mu_a = add_argument_node!(builder, name=:mu_a, typ=Float64, compute_grad=true)
    #one = add_constant_node!(builder, 1.)
    #a = add_random_choice_node!(builder, normal, inputs=[mu_a, one], addr=:a, typ=Float64, name=:a)
    #b = add_random_choice_node!(builder, normal, inputs=[a, one], addr=:b, typ=Float64, name=:b)
    #bar_val = add_gen_fn_call_node!(builder, bar, inputs=[a], addr=:bar, name=:bar_val)
    #c = add_julia_node!(builder, (a, b, bar) -> (a * b * bar), inputs=[a, b, bar_val], name=:c)
    #retval = add_random_choice_node!(builder, normal, inputs=[c, one], addr=:out, typ=Float64, name=:out)
    #set_return_node!(builder, retval)
    #ir = build_ir(builder)
    #eval(generate_generative_function(ir, :foo))
#
    #Gen.load_generated_functions()
#
    #function f(mu_a, a, b, z, out)
        #lpdf = 0.
        #mu_z = a
        #lpdf += logpdf(normal, z, mu_z, 1)
        #lpdf += logpdf(normal, a, mu_a, 1)
        #lpdf += logpdf(normal, b, a, 1)
        #c = a * b * (z + mu_z)
        #lpdf += logpdf(normal, out, c, 1)
        #return lpdf + 2 * out
    #end
#
    #mu_a = 1.
    #a = 2.
    #b = 3.
    #z = 4.
    #out = 5.
#
    ## get the initial trace
    #constraints = DynamicAssignment()
    #constraints[:a] = a
    #constraints[:b] = b
    #constraints[:out] = out
    #constraints[:bar => :z] = z
    #(trace, _) = initialize(foo, (mu_a,), constraints)
#
    ## compute gradients
    #selection = DynamicAddressSet()
    #push_leaf_node!(selection, :bar => :z)
    #push_leaf_node!(selection, :a)
    #push_leaf_node!(selection, :out)
    #selection = StaticAddressSet(selection)
    #retval_grad = 2.
    #((mu_a_grad,), value_trie, gradient_trie) = backprop_trace(foo, trace, selection, retval_grad)
#
    ## check value trie
    #@test get_leaf_node(value_trie, :a) == a
    #@test get_leaf_node(value_trie, :out) == out
    #@test get_leaf_node(value_trie, :bar => :z) == z
    #@test !has_leaf_node(value_trie, :b) # was not selected
    #@test length(get_internal_nodes(value_trie)) == 1
    #@test length(get_leaf_nodes(value_trie)) == 2
#
    ## check gradient trie
    #@test length(get_internal_nodes(gradient_trie)) == 1
    #@test length(get_leaf_nodes(gradient_trie)) == 2
    #@test !has_leaf_node(gradient_trie, :b) # was not selected
    #@test isapprox(mu_a_grad, finite_diff(f, (mu_a, a, b, z, out), 1, dx))
    #@test isapprox(get_leaf_node(gradient_trie, :a), finite_diff(f, (mu_a, a, b, z, out), 2, dx))
    #@test isapprox(get_leaf_node(gradient_trie, :out), finite_diff(f, (mu_a, a, b, z, out), 5, dx))
    #@test isapprox(get_leaf_node(gradient_trie, :bar => :z), finite_diff(f, (mu_a, a, b, z, out), 4, dx))
#end

end # @testset "static IR"
