using Gen

@gen function foo()
    n = @trace(uniform_discrete(1, 100), :n)
    total = 0.
    xs = Vector{Float64}(undef, n)
    for i=1:n
        xs[i] = @trace(normal(0, 1), (:x, i))
        total += xs[i]
    end
    @trace(normal(total, 1.), :obs)
    xs
end

ex = quote
@pkern function k1(tr, i::Int)
    mh(tr, select((:x, i)))[1]
end
end

println(macroexpand(Main, ex))

eval(ex)


@pkern function k2(tr, i::Int, j::Int)
    mh(tr, select((:x, i), (:x, j)))[1]
end

@pkern function k3(tr)
    tr = mh(tr, select(:n))[1]
    #tr = mh(tr, select(:obs))[1]
    tr
end

ex = quote
@kern function my_kernel()
    
    # cycle through each one
    for i in 1:@tr()[:n]
        @app k1(i)
    end

    # randomly pick one
    let i ~ uniform_discrete(1, @tr()[:n])
        @app k1(i)
    end

    # randomly pick two
    if @tr()[:n] > 10
        let i ~ uniform_discrete(1, @tr()[:n])
            let j ~ uniform_discrete(1, @tr()[:n])
                @app k2(i, j)
            end
        end
    end

    # change how many there are
    @app k3()
end
end


ex = quote
@kern function my_kernel()
    
    # cycle through each one
    for i in 1:@tr()[:n]
        @app k1(i)
    end

    # randomly pick one
    let i ~ uniform_discrete(1, @tr()[:n])
        @app k1(i)
    end

    # randomly pick two
    if @tr()[:n] > 10
        let i ~ uniform_discrete(1, @tr()[:n])
            let j ~ uniform_discrete(1, @tr()[:n])
                @app k2(i, j)
            end
        end
    end

    # change how many there are
    @app k3()
end
end


println(macroexpand(Main, ex))

eval(ex)

r = reversal(my_kernel)
println(r)

function do_inference(n::Int, iters::Int, check)
    obs = choicemap((:obs, 10))
    trace, = generate(foo, (), merge(obs, choicemap((:n, n))))
    for i=1:iters
        trace = my_kernel(trace, check, obs)
        trace = r(trace, check, obs)
    end
end

function do_inference_regular(n::Int, iters::Int)
    trace, = generate(foo, (), choicemap((:obs, 10), (:n, n)))
    for i=1:iters
        for j=1:trace[:n]
            trace = k1(trace, j)
        end
        i = uniform_discrete(1, trace[:n])
        trace = k1(trace, i)
    end
end


@time do_inference(100, 100, true)
@time do_inference(100, 100, true)

@time do_inference(100, 100, false)
@time do_inference(100, 100, false)

@time do_inference_regular(100, 100)
@time do_inference_regular(100, 100)
