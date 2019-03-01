using Gen

@gen (static) function foo()
    @param mu::Float64
    y = @trace(normal(mu, 1), :y)
    return y
end

load_generated_functions()

init_param!(foo, :mu, -1)

trace, = generate(foo, (), choicemap((:y, 3)))
step_size = 0.01
for iter=1:1000
    accumulate_param_gradients!(trace, 0.)
    grad_val = get_param_grad(foo, :mu)
    set_param!(foo, :mu, get_param(foo, :mu) + step_size * grad_val)
    zero_param_grad!(foo, :mu)
end

@assert abs(get_param(foo, :mu) - 3) < 1e-2
