#############################################################
# abstractions for constructing custom generative functions #
#############################################################

"""
    CustomDetGFTrace{T,S} <: Trace

Trace type for custom generative deterministic generative function.
"""
struct CustomDetGFTrace{T,S} <: Trace
    retval::T
    state::S
    args::Tuple
    gen_fn::Any
end

get_args(trace::CustomDetGFTrace) = trace.args

get_retval(trace::CustomDetGFTrace) = trace.retval

get_choices(trace::CustomDetGFTrace) = EmptyChoiceMap()

get_score(trace::CustomDetGFTrace) = 0.

project(trace::CustomDetGFTrace, selection::Selection) = 0.

get_gen_fn(trace::CustomDetGFTrace) = trace.gen_fn

"""
    CustomDetGF{T,S} <: GenerativeFunction{T,CustomDetGFTrace{T,S}} 

Abstract type for a custom deterministic generative function. 
"""
abstract type CustomDetGF{T,S} <: GenerativeFunction{T,CustomDetGFTrace{T,S}} end

"""
    retval, state = execute_det(gen_fn::CustomDetGF, args)

Execute the generative function and return the return value and the state.
"""
function execute_det end

"""
    state, retval, retdiff = update_det(gen_fn::CustomDetGF, state, args, argdiffs)

Update the arguments to the generative function and return new return value and state.
"""
function update_det end

"""
    arg_grads = gradient_det(gen_fn::CustomDetGF, state, retgrad) 

Return the gradient tuple with respect to the arguments.
"""
function gradient_det end

"""
    arg_grads = accumulate_param_gradients_get!(trace::CustomDefGF, retgrad, scaler)
    
Increment gradient accumulators for parameters the gradient with respect to the
arguments, optionally scaled, and return the gradient with respect to the
arguments (not scaled).

Given the previous state and a gradient with respect to the return value \$∇_y
J\$ (`retgrad`), return the following gradient (`arg_grads`) with respect to
the arguments \$x\$:
```math
∇_x J
```
Also increment the gradient accumulators for the trainable parameters \$Θ\$ of
the function by:
```math
\\mbox{scalar} * ∇_Θ J
"""
function accumulate_param_gradients_det! end

function simulate(gen_fn::CustomDetGF{T,S}, args::Tuple) where {T,S}
    retval, state = execute_det(gen_fn, args)
    CustomDetGFTrace{T,S}(retval, state, args, gen_fn)
end

function generate(gen_fn::CustomDetGF{T,S}, args::Tuple, choices::ChoiceMap) where {T,S}
    if !isempty(choices)
        error("Deterministic generative function makes no random choices")
    end
    retval, state = execute_det(gen_fn, args)
    trace = CustomDetGFTrace{T,S}(retval, state, args, gen_fn)
    trace, 0.
end

function update(trace::CustomDetGFTrace{T,S}, args::Tuple, argdiffs::Tuple, choices::ChoiceMap) where {T,S}
    if !isempty(choices)
        error("Deterministic generative function makes no random choices")
    end
    state, retval, retdiff = update_det(trace.gen_fn, trace.state, args, argdiffs)
    new_trace = CustomDetGFTrace{T,S}(retval, state, args, trace.gen_fn)
    (new_trace, 0., retdiff)
end

function regenerate(trace::CustomDetGFTrace, args::Tuple, argdiffs::Tuple, selection::Selection)
    update(trace, args, argdiffs, EmptyChoiceMap())
end

function choice_gradients(trace::CustomDetGFTrace, selection::Selection, retgrad)
    arg_grads = gradient_det(trace.gen_fn, trace.state, retgrad)
    (arg_grads, EmptyChoiceMap(), EmptyChoiceMap())
end

function accumulate_param_gradients!(trace::CustomDetGFTrace, retgrad, scaler)
    accumulate_param_gradients_det!(trace.gen_fn, trace.state, retgrad, scaler)
end

export CustomDetGF, CustomDetGFTrace, execute_det, update_det, gradient_det, accumulate_param_gradients_det!
