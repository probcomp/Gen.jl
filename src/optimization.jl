"""
    state = init(conf, gen_fn, param_list)

Get the initial state for an optimization algorithm applied to the given parameters.
"""
function init end

"""
    apply_update!(state)

Apply one update to the state.
"""
function apply_update! end

struct Optimizer
    states::Dict{GenerativeFunction,Any}
    conf::Any

    function Optimizer(conf, param_lists::Dict)
        states = Dict{GenerativeFunction,Any}()
            for (gen_fn, param_list) in param_lists
            states[gen_fn] = init(conf, gen_fn, param_list)
        end
        new(states, conf)
    end
end

function Optimizer(conf, gen_fn::GenerativeFunction)
    param_lists = Dict(gen_fn => collect(get_params(gen_fn)))
    Optimizer(conf, param_lists)
end


"""
    apply_update!(opt::Optimizer)

Perform one update step.
"""
function apply_update!(opt::Optimizer)
    for (_, state) in opt.states
        apply_update!(state)
    end
    nothing
end

"""
    conf = GradientDescentConf(step_size_init, step_size_beta)

Configuration for stochastic gradient descent update with step size given by `(t::Int) -> step_size_init * (step_size_beta + 1) / (step_size_beta + t)` where `t` is the iteration number.
"""
struct GradientDescentConf
    step_size_init::Float64
    step_size_beta::Float64
end

"""
    conf = ADAMConf(learning_rate, beta1, beta2)

Configuration for ADAM update.
"""
struct ADAMConf
    learning_rate::Float64
    beta1::Float64
    beta2::Float64
    epsilon::Float64
end

export ParameterSet, Optimizer, init, apply_update!
export GradientDescentConf, ADAMConf
