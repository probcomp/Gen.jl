using Gen
using Statistics: mean
using PyPlot
using Printf: @sprintf

# simple p
@gen function p()
    z = @trace(normal(0., 1.), :z)
    x = @trace(normal(z + 2., 0.3), :x)
    return x
end


# more sophisticated p that has multiple modes..
@gen function p()
    z = @trace(normal(0., 1.), :z)
    if z > 0.
        x = @trace(normal(z + 2., 0.1), :x)
    else
        if @trace(bernoulli(0.5), :m)
            x = @trace(normal(z + 1., 0.1), :x)
        else
            x = @trace(normal(z + 4., 0.1), :x)
        end
    end
    return x
end



sigmoid(x) = 1. / (1. + exp(-x))

option1(x, changept, sharpness) = sigmoid(sharpness * (x - changept))
option2(x, changept, sharpness) = 1 - option1(x, changept, sharpness)

# more sophsticated q with a noise that can vary in z
@gen function q(x::Float64)
    @param theta1::Float64
    @param theta2::Float64
    @param changept::Float64
    @param sharpness::Float64
    @param log_std1::Float64
    @param log_std2::Float64
    theta = theta1 * option1(x, changept, sharpness) + theta2 * option2(x, changept, sharpness)
    log_std = log_std1 * option1(x, changept, sharpness) + log_std2 * option2(x, changept, sharpness)
    z = @trace(normal(x + theta, exp(log_std)), :z)
    return z
end

# simple q
@gen function q(x::Float64)
    @param theta::Float64
    @param log_std::Float64
    z = @trace(normal(x + theta, exp(log_std)), :z)
    return z
end


@gen function q_batched(xs::Vector{Float64})
    @param theta::Float64
    @param log_std::Float64
    for i=1:length(xs)
        @trace(normal(xs[i] + theta, exp(log_std)), i => :z)
    end
end

function example_training_proc()
    init_param!(q, :theta, 0.)
    init_param!(q, :log_std, 0.)
    #init_param!(q, :sharpness, 1.)
    #init_param!(q, :theta1, 0.)
    #init_param!(q, :log_std1, 0.)
    #init_param!(q, :theta2, 0.)
    #init_param!(q, :log_std2, 0.)
    #init_param!(q, :changept, 1.)
    update = ParamUpdate(FixedStepGradientDescent(0.001), q)
    figure(figsize=(4,4))
    num = 1
    for iter=1:400#4000
        score = mean([lecture!(p, (), q, tr -> (tr[:x],)) for _=1:100])
        println("score: $score")#theta: $(get_param(q, :theta)), std: $(exp(get_param(q, :log_std)))")
        #println("score: $score, theta: $(get_param(q, :theta)), std: $(exp(get_param(q, :log_std)))")
        apply!(update)

        # plot
        if iter % 100 == 0 #1000 == 0
            subplot(2, 2, num)
            xs = range(-2, stop=4.5, length=200)
            zs = [q(x) for x in xs]
            scatter(xs, zs, marker=".", color="black")
            title("iter $iter")
            gca().set_xlim(-1, 5)
            gca().set_ylim(-2, 2)
            num += 1
        end
    end
    tight_layout()
    savefig("simple_p_stupid_q.png")
end

function example_training_proc_batched()
    init_param!(q_batched, :theta, 0.)
    init_param!(q_batched, :log_std, 0.)
    update = ParamUpdate(FixedStepGradientDescent(0.001), q_batched)
    for iter=1:100
        score = lecture_batched!(p, (), q_batched, trs -> (map(tr -> tr[:x], trs),), 100)
        println("score: $score, theta: $(get_param(q_batched, :theta)), std: $(exp(get_param(q_batched, :log_std)))")
        apply!(update)
    end
end

function plot_training_data()
    figure(figsize=(3,3))
    trs = [simulate(p, ()) for _=1:100]
    scatter([tr[:x] for tr in trs], [tr[:z] for tr in trs], marker=".", color="black")
    gca().set_xlim(-1, 5)
    gca().set_ylim(-2, 2)
    savefig("p_dist.png")
end

plot_training_data()

println("\nnot-batched")
example_training_proc()

#println("\nbatched")
#example_training_proc_batched()
