
#########################################
# gradient descent with fixed step size #
#########################################

mutable struct FixedStepGradientDescentDynamicDSLState
    step_size::Float64
    gen_fn::DynamicDSLFunction
    param_list::Vector
end

function init_update_state(conf::FixedStepGradientDescent, gen_fn::DynamicDSLFunction, param_list::Vector)
    FixedStepGradientDescentDynamicDSLState(conf.step_size, gen_fn, param_list)
end

function apply_update!(state::FixedStepGradientDescentDynamicDSLState)
    for param_name in state.param_list
        value = get_param(state.gen_fn, param_name)
        grad = get_param_grad(state.gen_fn, param_name)
        set_param!(state.gen_fn, param_name, value + grad * state.step_size)
        zero_param_grad!(state.gen_fn, param_name)
    end
end

####################
# gradient descent #
####################

mutable struct GradientDescentDynamicDSLState
    step_size_init::Float64
    step_size_beta::Float64
    gen_fn::DynamicDSLFunction
    param_list::Vector
    t::Int
end

function init_update_state(conf::GradientDescent, gen_fn::DynamicDSLFunction, param_list::Vector)
    GradientDescentDynamicDSLState(conf.step_size_init, conf.step_size_beta,
        gen_fn, param_list, 1)
end

function apply_update!(state::GradientDescentDynamicDSLState)
    step_size = state.step_size_init * (state.step_size_beta + 1) / (state.step_size_beta + state.t)
    for param_name in state.param_list
        value = get_param(state.gen_fn, param_name)
        grad = get_param_grad(state.gen_fn, param_name)
        set_param!(state.gen_fn, param_name, value + grad * step_size)
        zero_param_grad!(state.gen_fn, param_name)
    end
    state.t += 1
end
