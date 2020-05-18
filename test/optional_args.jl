#using Gen

@testset "optional positional args (calling + GFI)" begin

    @gen function foo(x, y=2, (grad)(z::Float64=3.0))
        @param theta::Float64
        a = @trace(normal(x+y+z, 1), :a)
        b = @trace(normal(a+theta, 1), :b)
        return (x, y, z, x+y+z)
    end

    # initialize theta to zero for non-gradient tests
    init_param!(foo, :theta, 0.)

    # test directly calling with varying args
    @test foo(1) == (1, 2, 3, 6)
    @test foo(5, 4) == (5, 4, 3, 12)
    @test foo(2, 3, 4) == (2, 3, 4, 9)

    # test simulate using default args
    tr = simulate(foo, (1,))
    @test get_args(tr) == (1, 2, 3)
    @test get_retval(tr) == (1, 2, 3, 6)

    # test generate using default args
    tr, w = generate(foo, (1,), choicemap(:a => 5))
    @test get_args(tr) == (1, 2, 3)
    @test isapprox(w, logpdf(normal, 5, 1+2+3, 1))

    # test update with varying args and argdiffs
    new_tr, w_diff, _, _ = update(tr, (2, 3, 4),
        (UnknownChange(),  UnknownChange(), UnknownChange()), choicemap())
    @test get_args(new_tr) == (2, 3, 4)
    @test new_tr[:a] == tr[:a]
    @test new_tr[:b] == tr[:b]
    @test isapprox(w_diff, logpdf(normal, new_tr[:a], 2+3+4, 1) -
                           logpdf(normal, tr[:a], 1+2+3, 1))

    new_tr, w_diff, _, _ = update(tr, (2,),
        (UnknownChange(),), choicemap(:a => 6))
    @test get_args(new_tr) == (2, 2, 3)
    @test new_tr[:a] == 6
    @test new_tr[:b] == tr[:b]
    @test isapprox(w_diff, (logpdf(normal, new_tr[:a], 2+2+3, 1) +
                            logpdf(normal, new_tr[:b], new_tr[:a], 1)) -
                           (logpdf(normal, tr[:a], 1+2+3, 1) +
                            logpdf(normal, tr[:b], tr[:a], 1)))

    # test regenerate with varying args and argdiffs
    new_tr, w_diff, _ = regenerate(tr, (2, 3, 4),
        (UnknownChange(),  UnknownChange(), UnknownChange()), select(:b))
    @test get_args(new_tr) == (2, 3, 4)
    @test new_tr[:a] == tr[:a]
    expected_score = (logpdf(normal, new_tr[:a], 2+3+4, 1) +
                      logpdf(normal, new_tr[:b], new_tr[:a], 1))
    @test isapprox(get_score(new_tr), expected_score)
    expected_diff = (logpdf(normal, new_tr[:a], 2+3+4, 1) -
                     logpdf(normal, tr[:a], 1+2+3, 1))
    @test isapprox(w_diff, expected_diff)

    new_tr, w_diff, _ = regenerate(tr, (2, 3),
        (UnknownChange(), UnknownChange()), select(:b))
    @test get_args(new_tr) == (2, 3, 3)
    @test new_tr[:a] == tr[:a]
    expected_score = (logpdf(normal, new_tr[:a], 2+3+3, 1) +
                      logpdf(normal, new_tr[:b], new_tr[:a], 1))
    @test isapprox(get_score(new_tr), expected_score)
    expected_diff = (logpdf(normal, new_tr[:a], 2+3+3, 1) -
                     logpdf(normal, tr[:a], 1+2+3, 1))
    @test isapprox(w_diff, expected_diff)

    # test choice_gradients and accumulate_param_gradients!
    function foo_lpdf(x, y, z, a, b)
        lpdf = 0.
        lpdf += logpdf(normal, a, x+y+z, 1)
        lpdf += logpdf(normal, b, a, 1)
        return lpdf
    end

    tr, _ = generate(foo, (2,), choicemap(:a => 5, :b => 6))
    arg_grads = accumulate_param_gradients!(tr, nothing)
    @test length(arg_grads) == 3 # return gradients for all args, incl. optional
    _, _, z_grad = arg_grads
    @test isapprox(z_grad, finite_diff(foo_lpdf, (2, 2, 3, 5, 6), 3, dx))

    arg_grads, _, _ = choice_gradients(tr, select(:a), nothing)
    @test length(arg_grads) == 3 # return gradients for all args, incl. optional
    _, _, z_grad = arg_grads
    @test isapprox(z_grad, finite_diff(foo_lpdf, (2, 2, 3, 5, 6), 3, dx))

    # test nested calls
    @gen function bar()
        return @trace(foo(1), :foo)
    end
    @test bar() == (1, 2, 3, 6)

    tr, w = generate(bar, (), choicemap((:foo => :a) => 5))
    @test get_args(tr) == ()
    sub_tr = Gen.get_call(tr, :foo).subtrace
    @test get_args(sub_tr) == (1, 2, 3) # args of subtrace should have defaults
    @test get_score(tr) == get_score(sub_tr)
    @test isapprox(w, logpdf(normal, 5, 1+2+3, 1))

    # test optional GenerativeFunction arguments
    @gen function outer(inner::GenerativeFunction=foo)
        @trace(inner(1), :inner)
    end

    @test outer() == (1, 2, 3, 6)
    tr, w = generate(outer, (), choicemap((:inner => :a) => 5))
    @test get_args(tr) == (foo,)
    sub_tr = Gen.get_call(tr, :inner).subtrace
    @test get_args(sub_tr) == (1, 2, 3) # args of subtrace should have defaults
    @test get_score(tr) == get_score(sub_tr)
    @test isapprox(w, logpdf(normal, 5, 1+2+3, 1))

    @gen function foo2(x)
        @trace(normal(x, 1), :a)
        return x
    end
    @test outer(foo2) == 1
    tr, w = generate(outer, (foo2,), choicemap((:inner => :a) => 5))
    @test get_args(tr) == (foo2,)
    sub_tr = Gen.get_call(tr, :inner).subtrace
    @test get_args(sub_tr) == (1,)
    @test get_score(tr) == get_score(sub_tr)
    @test isapprox(w, logpdf(normal, 5, 1, 1))

end

@testset "optional positional args (combinators)" begin

    @gen function foo(t::Int, y_prev::Bool, z1::Float64=0.25, z2::Float64=0.75)
        y = @trace(bernoulli(y_prev ? z1 : z2), :y)
    end

    # test call_at
    at_foo = call_at(foo, Int)
    addr = 3
    tr, w = generate(at_foo, (0, true, addr), choicemap((addr => :y) => true))
    @test tr[addr => :y] == true
    @test isapprox(w, logpdf(bernoulli, tr[addr => :y], 0.25))
    tr, w = generate(at_foo, (0, true, 0.5, addr), choicemap((addr => :y) => true))
    @test tr[addr => :y] == true
    @test isapprox(w, logpdf(bernoulli, tr[addr => :y], 0.5))

    # test map
    map_foo = Map(foo)
    addr = 1
    tr, w = generate(map_foo,
        ([1, 2, 3], [false, true, false]),
        choicemap((addr => :y) => true))
    @test tr[addr => :y] == true
    @test isapprox(w, logpdf(bernoulli, tr[addr => :y], 0.75))
    tr, w = generate(map_foo,
        ([1, 2, 3], [true, false, true], [0.5, 0.25, 0.0]),
        choicemap((addr => :y) => true))
    @test tr[addr => :y] == true
    @test isapprox(w, logpdf(bernoulli, tr[addr => :y], 0.5))

    # test unfold
    unfold_foo = Unfold(foo)
    addr = 1
    tr, w = generate(unfold_foo, (3, false),
        choicemap((addr => :y) => true))
    @test tr[addr => :y] == true
    @test isapprox(w, logpdf(bernoulli, tr[addr => :y], 0.75))
    tr, w = generate(unfold_foo, (3, false, 0.05, 0.95),
        choicemap((addr => :y) => true))
    @test tr[addr => :y] == true
    @test isapprox(w, logpdf(bernoulli, tr[addr => :y], 0.95))

    # test recurse
    @gen function production(depth::Int, branches::Int=1)
        n_children = @trace(bernoulli(0.5), :branch) ? branches : 0
        return Production(depth, [depth+1 for i=1:n_children])
    end

    @gen function aggregation(depth::Int, child_outputs::Vector, combiner=sum)
        if length(child_outputs) == 0
            out = depth
        else
            out = combiner(child_outputs)
        end
    end

    bar1 = Recurse(production, aggregation, 1, Int, Int, Any)
    constraints = choicemap()
    constraints[(1, Val(:production)) => :branch] = true
    constraints[(2, Val(:production)) => :branch] = false
    tr, w = generate(bar1, (1, 1), constraints)
    @test get_retval(tr) == 2
    constraints[(2, Val(:production)) => :branch] = true
    tr, w = zip([generate(bar1, (1, 1), constraints) for i in 1:100]...)
    @test all(get_retval.(tr) .>= 2)

    @gen new_prod(d::Int) =
        @trace(production(d, 2))
    @gen new_aggr(d::Int, cvals::Vector) =
        @trace(aggregation(d, cvals, xs -> join(string.(xs))))

    bar2 = Recurse(new_prod, new_aggr, 2, Int, Int, Any)
    constraints = choicemap()
    constraints[(1, Val(:production)) => :branch] = true
    constraints[(Gen.get_child(1, 1, 2), Val(:production)) => :branch] = false
    constraints[(Gen.get_child(1, 2, 2), Val(:production)) => :branch] = false
    tr, w = generate(bar2, (1, 1), constraints)
    @test get_retval(tr) == "22"
    constraints[(Gen.get_child(1, 1, 2), Val(:production)) => :branch] = true
    constraints[(Gen.get_child(1, 2, 2), Val(:production)) => :branch] = true
    constraints[(Gen.get_child(2, 1, 2), Val(:production)) => :branch] = false
    constraints[(Gen.get_child(2, 2, 2), Val(:production)) => :branch] = false
    constraints[(Gen.get_child(3, 1, 2), Val(:production)) => :branch] = false
    constraints[(Gen.get_child(3, 2, 2), Val(:production)) => :branch] = false
    tr, w = generate(bar2, (1, 1), constraints)
    @test get_retval(tr) == "3333"
end
