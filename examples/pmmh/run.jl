include("model.jl")

load_generated_functions()

# particle marginal metropolis-hastings

@gen function var_proposal()
    var_x = @read(:var_x)
    var_y = @read(:var_y)
    #@addr(normal(var_x, sqrt(0.15)), :var_x)
    #@addr(normal(var_y, sqrt(0.08)), :var_y)
    @addr(normal(var_x, sqrt(0.5)), :var_x)
    @addr(normal(var_y, sqrt(0.5)), :var_y)
end

@gen function observer(ys)
    for (i, y) in enumerate(ys)
        @addr(dirac(y), :hmm => :y => i)
    end
end

function initial_collapsed_trace(ys)
    T = length(ys)
    constraints = get_choices(simulate(observer, (ys,)))
    (trace, weight) = generate(model_collapsed, (T,), constraints)
    trace
end

import Random
Random.seed!(1)

# generate synthetic dataset
T = 500
(xs_sim, ys_sim) = hmm(10., 1., T)

# do inference
function do_inference()
    trace = initial_collapsed_trace(ys_sim)
    for iter=1:1000
        score = get_call_record(trace).score
        println("score: $score")
        trace = mh(model_collapsed, var_proposal, (), trace)
        choices = get_choices(trace)
	    println("var_x: $(choices[:var_x]), var_y: $(choices[:var_y])")
    end
end

do_inference()
