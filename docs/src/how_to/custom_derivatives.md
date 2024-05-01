# How to Write Custom Gradients

To add a custom gradient for a differentiable deterministic computation, define a concrete subtype of [`CustomGradientGF`](@ref) with the following methods:

- [`apply`](@ref)

- [`gradient`](@ref)

- [`has_argument_grads`](@ref)

For example:

```julia
struct MyPlus <: CustomGradientGF{Float64} end

Gen.apply(::MyPlus, args) = args[1] + args[2]
Gen.gradient(::MyPlus, args, retval, retgrad) = (retgrad, retgrad)
Gen.has_argument_grads(::MyPlus) = (true, true)
```

# [Optimizing Trainable Parameters](@id optimizing-internal)

To add support for a new type of gradient-based parameter update, create a new type with the following methods defined for the types of generative functions that are to be supported.
- [Gen.init_update_state](@ref)
- [Gen.apply_update!](@ref)
```
