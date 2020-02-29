using Gen

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
