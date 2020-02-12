using Gen
import Random
import MacroTools

@gen function model()
    n = @trace(geometric(0.5), :n)
    total = 0.
    for i=1:n
        total += @trace(normal(0, 1), (:x, i))
    end
    @trace(normal(total, 1.), :y)
    total
end

@gen function random_walk_proposal(trace, i::Int)
    @trace(normal(trace[(:x, i)], 0.1), (:x, i))
end

@pkern function k1(trace, i::Int)
    trace, = mh(trace, random_walk_proposal, (i,))
    trace
end

@gen function add_remove_proposal(trace)
    n = trace[:n]
    total = get_retval(trace)
    add = (n == 0) || @trace(bernoulli(0.5), :add)
    if add
        @trace(normal(trace[:y] - total, 1.), :new_x)
    end
    (n, add)
end

function add_remove_involution(trace, fwd_choices, ret, args)
    (n, add) = ret
    bwd_choices = choicemap()
    new_n = add ? n + 1 : n - 1
    constraints = choicemap((:n, new_n))
    if add 
        bwd_choices[:add] = false
        constraints[(:x, new_n)] = fwd_choices[:new_x]
    else
        bwd_choices[:new_x] = trace[(:x, n)]
        (new_n > 0) && (bwd_choices[:add] = true)
    end
    new_trace, weight, = update(trace, (), (), constraints)
    (new_trace, bwd_choices, weight)
end

@pkern function k2(trace)
    trace, = mh(trace, add_remove_proposal, (), add_remove_involution, check_round_trip=true)
    trace
end

@pkern function k3(trace)
    perm = Random.randperm(trace[:n])
    constraints = choicemap()
    for (i, j) in enumerate(perm)
        constraints[(:x, i)] = trace[(:x, j)]
        constraints[(:x, j)] = trace[(:x, i)]
    end
    trace, = update(trace, (), (), constraints)
    trace
end

@rkern k1 : k1
@rkern k2 : k2
@rkern k3 : k3

max_n_add_remove = 10 # to test that we have escaped the body of the ckern properly

ex = quote
@ckern function my_kernel((@T))
    
    # cycle through the x's and do a random walk update on each one
    for i in 1:(@T)[:n]
        (@T) ~ k1((@T), i)
    end

    # repeatedly pick a random x and do a random walk update on it
    if (@T)[:n] > 0
        for rep in 1:10
            let i ~ uniform_discrete(1, (@T)[:n])
                (@T) ~ k1((@T), i)
            end
        end
    end

    # remove the last x, or add a new one, a random number of times
    let n_add_remove_reps ~ uniform_discrete(0, max_n_add_remove)
        for rep in 1:n_add_remove_reps
            (@T) ~ k2((@T))
        end
    end

    # permute the x's
    (@T) ~ k3((@T))
end
end

#println(MacroTools.striplines(macroexpand(Main, MacroTools.postwalk(MacroTools.unblock, ex))))
eval(ex)

function run_dsl_kernel(n::Int, iters::Int, check)
    obs = choicemap((:y, 10))
    trace, = generate(model, (), merge(obs, choicemap((:n, n))))
    for i=1:iters
        trace = my_kernel(trace, check, obs)
    end
end

function run_reversal_dsl_kernel(n::Int, iters::Int, check)
    obs = choicemap((:y, 10))
    trace, = generate(model, (), merge(obs, choicemap((:n, n))))
    for i=1:iters
        trace = reversal(my_kernel)(trace, check, obs)
    end
end


# checks disabled
@time run_dsl_kernel(100, 100, false)
@time run_dsl_kernel(100, 100, false)

# checks enabled
@time run_dsl_kernel(100, 100, true)
@time run_dsl_kernel(100, 100, true)

# run reverse DSL with checks enabled
@time run_reversal_dsl_kernel(100, 100, true)
@time run_reversal_dsl_kernel(100, 100, true)
