"""
    set_param!(gen_fn::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol, value)

Set the value of a trainable parameter of the generative function.

NOTE: Does not update the gradient accumulator value.
"""
function set_param!(gf::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol, value)
    gf.params[name] = value
end

"""
    value = get_param(gen_fn::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol)

Get the current value of a trainable parameter of the generative function.
"""
function get_param(gf::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol)
    gf.params[name]
end

"""
    value = get_param_grad(gen_fn::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol)

Get the current value of the gradient accumulator for a trainable parameter of the generative function.
"""
function get_param_grad(gf::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol)
    gf.params_grad[name]
end

"""
    zero_param_grad!(gen_fn::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol)

Reset the gradient accumlator for a trainable parameter of the generative function to all zeros.
"""
function zero_param_grad!(gf::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol)
    gf.params_grad[name] = zero(gf.params[name])
end

"""
    set_param_grad!(gen_fn::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol, grad_value)

Set the gradient accumlator for a trainable parameter of the generative function.
"""
function set_param_grad!(gf::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol, grad_value)
    gf.params_grad[name] = grad_value
end

"""
    init_param!(gen_fn::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol, value)

Initialize the the value of a named trainable parameter of a generative function.

Also generates the gradient accumulator for that parameter to `zero(value)`.

Example:
```julia
init_param!(foo, :theta, 0.6)
```
"""
function init_param!(gf::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol, value)
    set_param!(gf, name, value)
    zero_param_grad!(gf, name)
end

get_params(gf::Union{DynamicDSLFunction,StaticIRGenerativeFunction}) = keys(gf.params)

export set_param!, get_param, get_param_grad, zero_param_grad!, set_param_grad!, init_param!

#########################################
# gradient descent with fixed step size #
#########################################

mutable struct FixedStepGradientDescentBuiltinDSLState
    step_size::Float64
    gen_fn::Union{DynamicDSLFunction,StaticIRGenerativeFunction}
    param_list::Vector
end

function init_update_state(conf::FixedStepGradientDescent,
        gen_fn::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, param_list::Vector)
    FixedStepGradientDescentBuiltinDSLState(conf.step_size, gen_fn, param_list)
end

function apply_update!(state::FixedStepGradientDescentBuiltinDSLState)
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

mutable struct GradientDescentBuiltinDSLState
    step_size_init::Float64
    step_size_beta::Float64
    gen_fn::Union{DynamicDSLFunction,StaticIRGenerativeFunction}
    param_list::Vector
    t::Int
end

function init_update_state(conf::GradientDescent,
        gen_fn::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, param_list::Vector)
    GradientDescentBuiltinDSLState(conf.step_size_init, conf.step_size_beta,
        gen_fn, param_list, 1)
end

function apply_update!(state::GradientDescentBuiltinDSLState)
    step_size = state.step_size_init * (state.step_size_beta + 1) / (state.step_size_beta + state.t)
    for param_name in state.param_list
        value = get_param(state.gen_fn, param_name)
        grad = get_param_grad(state.gen_fn, param_name)
        set_param!(state.gen_fn, param_name, value + grad * step_size)
        zero_param_grad!(state.gen_fn, param_name)
    end
    state.t += 1
end
