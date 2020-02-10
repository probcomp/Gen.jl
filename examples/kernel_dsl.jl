using Gen

# a simple model

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

# declare three primitive stationary kernels

@pkern function k1(tr, i::Int)
    mh(tr, select((:x, i)))[1]
end

@revkern k1 : k1

println(reversal(k1))

@pkern function k2(tr, i::Int, j::Int)
    mh(tr, select((:x, i), (:x, j)))[1]
end

@revkern k2 : k2

@pkern function k3(tr)
    mh(tr, select(:n))[1]
end

@revkern k3 : k3

# define a composite stationary kernel

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

function run_dsl_kernel(n::Int, iters::Int, check)
    obs = choicemap((:obs, 10))
    trace, = generate(foo, (), merge(obs, choicemap((:n, n))))
    for i=1:iters
        trace = my_kernel(trace, check, obs)
    end
end

function run_reversal_dsl_kernel(n::Int, iters::Int, check)
    obs = choicemap((:obs, 10))
    trace, = generate(foo, (), merge(obs, choicemap((:n, n))))
    for i=1:iters
        trace = reversal(my_kernel)(trace, check, obs)
    end
end


function run_regular_kernel(n::Int, iters::Int)
    trace, = generate(foo, (), choicemap((:obs, 10), (:n, n)))
    for i=1:iters

        # cycle through each one
        for j=1:trace[:n]
            trace = k1(trace, j)
        end

        # randomly pick one
        i = uniform_discrete(1, trace[:n])
        trace = k1(trace, i)

        # randomly pick two
        if trace[:n] > 10
            i = uniform_discrete(1, trace[:n])
            j = uniform_discrete(1, trace[:n])
            trace = k2(trace, i, j)
        end
    end
end

# checks disabled
@time run_dsl_kernel(100, 100, false)
@time run_dsl_kernel(100, 100, false)

# checks enabled
@time run_dsl_kernel(100, 100, true)
@time run_dsl_kernel(100, 100, true)

# plain Julia composition
@time run_regular_kernel(100, 100)
@time run_regular_kernel(100, 100)

# run reverse DSL with checks enabled
@time run_reversal_dsl_kernel(100, 100, true)
@time run_reversal_dsl_kernel(100, 100, true)
