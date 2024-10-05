# [Customizing Gradients](@id custom_gradients_howto)

## Determistic Functions with Custom Gradients

To add a custom gradient for a differentiable deterministic computation, define a concrete subtype of [`CustomGradientGF`](@ref) with the following methods:

- [`apply`](@ref)

- [`gradient`](@ref)

- [`has_argument_grads`](@ref)

For example, we can implement binary addition with a manually-defined gradient:

```julia
struct MyPlus <: CustomGradientGF{Float64} end

Gen.apply(::MyPlus, args) = args[1] + args[2]
Gen.gradient(::MyPlus, args, retval, retgrad) = (retgrad, retgrad)
Gen.has_argument_grads(::MyPlus) = (true, true)
```

## Customizing Parameter Updates

To add support for a new type of gradient-based parameter update, create a new [parameter update configuration](@ref update_configurations) with the following methods defined for the types of generative functions that are to be supported.

- [`Gen.init_update_state`](@ref)
- [`Gen.apply_update!`](@ref)

As an example, the built-in update configuration, [`FixedStepGradientDescent`](@ref), is implemented as follows:

```julia
struct FixedStepGradientDescent
    step_size::Float64
end

mutable struct FixedStepGradientDescentBuiltinDSLState
    step_size::Float64
    gen_fn::Union{Gen.DynamicDSLFunction,Gen.StaticIRGenerativeFunction}
    param_list::Vector
end

function Gen.init_update_state(conf::FixedStepGradientDescent,
        gen_fn::Union{Gen.DynamicDSLFunction,Gen.StaticIRGenerativeFunction}, param_list::Vector)
    FixedStepGradientDescentBuiltinDSLState(conf.step_size, gen_fn, param_list)
end

function Gen.apply_update!(state::FixedStepGradientDescentBuiltinDSLState)
    for param_name in state.param_list
        value = Gen.get_param(state.gen_fn, param_name)
        grad = Gen.get_param_grad(state.gen_fn, param_name)
        Gen.set_param!(state.gen_fn, param_name, value + grad * state.step_size)
        Gen.zero_param_grad!(state.gen_fn, param_name)
    end
end
```
