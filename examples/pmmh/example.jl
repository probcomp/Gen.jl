using Gen

include("pf.jl")

function x_mean(x_prev::Real, t::Int)
    (x_prev / 2.) + 25 * (x_prev / (1 + x_prev * x_prev)) + 8 * cos(1.2 * t)
end

y_mean(x::Real) = (x * x / 20.)

@gen function init()
    return @trace(normal(0, 5), :x)
end

@gen function dynamics(t::Int, prev_state::Float64, var_x::Float64)
    state = @trace(normal(x_mean(prev_state, t), sqrt(var_x)), :x)
    return state
end

@gen function emission(t::Int, state::Float64, var_y::Float64)
    return @trace(normal(y_mean(state), sqrt(var_y)), :y)
end

collapsed_hmm = ParticleFilterCombinator(init, dynamics, emission, 100)

@gen function model()
    var_x::Float64 = exp(@trace(normal(0, 2), :var_x))
    var_y::Float64 = exp(@trace(normal(0, 2), :var_y))
    @trace(collapsed_hmm(T, (), (var_x,), (var_y,)), :hmm)
end

@gen function var_x_proposal(prev)
    var_x::Float64 = get_choices(prev)[:var_x]
    @trace(normal(var_x, sqrt(0.5)), :var_x)
end

@gen function var_y_proposal(prev)
    var_y::Float64 = get_choices(prev)[:var_y]
    @trace(normal(var_y, sqrt(0.5)), :var_y)
end

# generate synthetic dataset
T = 100
var_x = 4.
var_y = 1.
xs = Vector{Float64}(undef, T)
ys = Vector{Float64}(undef, T)
xs[1] = init()
ys[1] = emission(1, xs[1], var_y)
for t=2:T
    xs[t] = dynamics(t, xs[t-1], var_x)
    ys[t] = emission(t, xs[t], var_y)
end
println(ys)

observations = choicemap()
for t=1:T
    observations[:hmm => t => :y] = ys[t]
end

# do inference
function do_inference(n)
    tr, _ = generate(model, (), observations)
    for iter=1:n

        score = get_score(tr)
        println("score: $score")

        (tr, _) = mh(tr, select(:var_x))
        (tr, _) = mh(tr, select(:var_y))
        (tr, _) = mh(tr, var_x_proposal, ())
        (tr, _) = mh(tr, var_y_proposal, ())

        choices = get_choices(tr)
	    println("var_x: $(exp(choices[:var_x])), var_y: $(exp(choices[:var_y]))")
    end
end

do_inference(1000)
