"""
    state = init_update_state(conf, gen_fn::GenerativeFunction, param_list::Vector)

Get the initial state for a parameter update to the given parameters of the given generative function.

`param_list` is a vector of references to parameters of `gen_fn`.
`conf` configures the update.
"""
function init_update_state end

"""
    apply_update!(state)

Apply one parameter update, mutating the values of the trainable parameters, and possibly also the given state.
"""
function apply_update! end

"""
    update = ParamUpdate(conf, param_lists...)

Return an update configured by `conf` that applies to set of parameters defined by `param_lists`.

Each element in `param_lists` value is is pair of a generative function and a vector of its parameter references.

**Example**. To construct an update that applies a gradient descent update to the parameters `:a` and `:b` of generative function `foo` and the parameter `:theta` of generative function `:bar`:

```julia
update = ParamUpdate(GradientDescent(0.001, 100), foo => [:a, :b], bar => [:theta])
```

------------------------------------------------------------------------------------------
Syntactic sugar for the constructor form above.

    update = ParamUpdate(conf, gen_fn::GenerativeFunction)

Return an update configured by `conf` that applies to all trainable parameters owned by the given generative function.

Note that trainable parameters not owned by the given generative function will not be updated, even if they are used during execution of the function.

**Example**. If generative function `foo` has parameters `:a` and `:b`, to construct an update that applies a gradient descent update to the parameters `:a` and `:b`:

```julia
update = ParamUpdate(GradientDescent(0.001, 100), foo)
```
"""
struct ParamUpdate
    states::Dict{GenerativeFunction,Any}
    conf::Any
    function ParamUpdate(conf, param_lists...)
        states = Dict{GenerativeFunction,Any}()
        for (gen_fn, param_list) in param_lists
            states[gen_fn] = init_update_state(conf, gen_fn, param_list)
        end
        new(states, conf)
    end
    function ParamUpdate(conf, gen_fn::GenerativeFunction)
        param_lists = Dict(gen_fn => collect(get_params(gen_fn)))
        ParamUpdate(conf, param_lists...)
    end
end


"""
    apply!(update::ParamUpdate)

Perform one step of the update.
"""
function apply!(update::ParamUpdate)
    for (_, state) in update.states
        apply_update!(state)
    end
    nothing
end

"""
    conf = FixedStepGradientDescent(step_size)

Configuration for stochastic gradient descent update with fixed step size.
"""
struct FixedStepGradientDescent
    step_size::Float64
end

"""
    conf = GradientDescent(step_size_init, step_size_beta)

Configuration for stochastic gradient descent update with step size given by `(t::Int) -> step_size_init * (step_size_beta + 1) / (step_size_beta + t)` where `t` is the iteration number.
"""
struct GradientDescent
    step_size_init::Float64
    step_size_beta::Float64
end

"""
    conf = ADAM(learning_rate, beta1, beta2, epsilon)

Configuration for ADAM update.
"""
struct ADAM
    learning_rate::Float64
    beta1::Float64
    beta2::Float64
    epsilon::Float64
end

export ParameterSet, ParamUpdate, apply!
export FixedStepGradientDescent, GradientDescent, ADAM
