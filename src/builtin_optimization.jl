#############################

# primitives for in-place gradient accumulation

function in_place_add!(param::Array, increment::Array, scale_factor::Real)
    # NOTE: it ignores the scale_factor, because it is not a parameter...
    # scale factors only affect parameters
    # TODO this is potentially very confusing!
    @simd for i in 1:length(param)
        param[i] += increment[i]
    end
    return param
end

function in_place_add!(param::Array, increment::Array)
    @inbounds @simd for i in 1:length(param)
        param[i] += increment[i]
    end
    return param
end

function in_place_add!(param::Real, increment::Real, scale_factor::Real)
    return param + increment
end

function in_place_add!(param::Real, increment::Real)
    return param + increment
end

mutable struct ThreadsafeAccumulator{T}
    value::T
    lock::ReentrantLock
end

ThreadsafeAccumulator(value) = ThreadsafeAccumulator(value, ReentrantLock())

# TODO not threadsafe
function get_current_value(accum::ThreadsafeAccumulator)
    return accum.value
end

function in_place_add!(param::ThreadsafeAccumulator{Real}, increment::Real, scale_factor::Real)
    lock(param.lock)
    try
        param.value = param.value + increment * scale_factor
    finally
        unlock(param.lock)
    end
    return param
end

function in_place_add!(param::ThreadsafeAccumulator{Real}, increment::Real)
    lock(param.lock)
    try
        param.value = param.value + increment
    finally
        unlock(param.lock)
    end
    return param
end

function in_place_add!(param::ThreadsafeAccumulator{<:Array}, increment, scale_factor::Real)
    lock(param.lock)
    try
        @simd for i in 1:length(param.value)
            param.value[i] += increment[i] * scale_factor
        end
    finally
        unlock(param.lock)
    end
    return param
end

function in_place_add!(param::ThreadsafeAccumulator{<:Array}, increment)
    lock(param.lock)
    try
        @simd for i in 1:length(param.value)
            param.value[i] += increment[i]
        end
    finally
        unlock(param.lock)
    end
    return param
end

#############################



"""
    set_param!(gen_fn::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol, value)

Set the value of a trainable parameter of the generative function.

NOTE: Does not update the gradient accumulator value.
"""
function set_param!(gf::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol, value)
    return gf.params[name] = value
end

"""
    value = get_param(gen_fn::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol)

Get the current value of a trainable parameter of the generative function.
"""
function get_param(gf::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol)
    return gf.params[name]
end

"""
    value = get_param_grad(gen_fn::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol)

Get the current value of the gradient accumulator for a trainable parameter of the generative function.

Not threadsafe.
"""
function get_param_grad(gf::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol)
    try
	val = gf.params_grad[name] # the accumulator
        return get_current_value(val)
    catch KeyError
        error("parameter $name not found")
    end
    return val
end

"""
    zero_param_grad!(gen_fn::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol)

Reset the gradient accumlator for a trainable parameter of the generative function to all zeros.

Not threadsafe.
"""
function zero_param_grad!(gf::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol)
    gf.params_grad[name] = ThreadsafeAccumulator(zero(gf.params[name])) # TODO avoid allocation?
    return gf.params_grad[name]
end

"""
    set_param_grad!(gen_fn::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol, grad_value)

Set the gradient accumlator for a trainable parameter of the generative function.

Not threadsafe.
"""
function set_param_grad!(gf::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol, grad_value)
    gf.params_grad[name] = ThreadsafeAccumulator(grad_value)
    return grad_value
end

# TODO document me; it is threadsafe..
function increment_param_grad!(gf::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol, increment, scale_factor)
    in_place_add!(gf.params_grad[name], increment, scale_factor)
end

# TODO document me; it is threadsafe..
function increment_param_grad!(gf::Union{DynamicDSLFunction,StaticIRGenerativeFunction}, name::Symbol, increment)
    in_place_add!(gf.params_grad[name], increment)
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

export set_param!, get_param, get_param_grad, zero_param_grad!, set_param_grad!, init_param!, increment_param_grad!

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
