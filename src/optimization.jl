import Parameters

# TODO we should modify the semantics of the log probability contribution to
# the gradient so that everything is gradient descent instead of ascent. this
# will also fix the misnomer names
#
# TODO add tests specifically for JuliaParameterStore etc.

export in_place_add!

export FixedStepGradientDescent
export DecayStepGradientDescent
export init_optimizer
export apply_update!

export JuliaParameterStore
export init_parameter!
export increment_gradient!
export reset_gradient!
export get_parameter_value
export get_gradient
export JULIA_PARAMETER_STORE_KEY

export default_julia_parameter_store
export default_parameter_context

############################
# optimizer specifications #
############################

"""
    conf = FixedStepGradientDescent(step_size)

Configuration for stochastic gradient descent update with fixed step size.
"""
Parameters.@with_kw struct FixedStepGradientDescent
    step_size::Float64
end

"""
    conf = GradientDescent(step_size_init, step_size_beta)

Configuration for stochastic gradient descent update with step size given by `(t::Int) -> step_size_init * (step_size_beta + 1) / (step_size_beta + t)` where `t` is the iteration number.
"""
Parameters.@with_kw struct DecayStepGradientDescent
    step_size_init::Float64
    step_size_beta::Float64
end

# TODO add gradient descent with momentum
# TODO add update

#################
# in_place_add! #
#################

function in_place_add! end

function in_place_add!(value::Array, increment)
    @simd for i in 1:length(value)
        value[i] += increment[i]
    end
    return value
end

# this exists so user can use the same function on scalars and arrays
function in_place_add!(value::Real, increment::Real)
    return value + increment
end


###########################
# thread-safe accumulator # 
###########################

mutable struct Accumulator{T<:Union{Real,Array}}
    value::T
    lock::ReentrantLock
end

Accumulator(value) = Accumulator(value, ReentrantLock())

# NOTE: not thread-safe because it may return a reference to the Array
get_value(accum::Accumulator) = accum.value

function fill_with_zeros!(accum::Accumulator{T}) where {T <: Real}
    lock(accum.lock)
    try
        accum.value = zero(T)
    finally
        unlock(accum.lock)
    end
    return accum
end

function fill_with_zeros!(accum::Accumulator{<:Array{T}}) where {T}
    lock(accum.lock)
    try
        fill!(accum.value, zero(T))
    finally
        unlock(accum.lock)
    end
    return accum
end

function in_place_add!(accum::Accumulator{<:Real}, increment::Real, scale_factor::Real)
    lock(accum.lock)
    try
        accum.value = accum.value + increment * scale_factor
    finally
        unlock(accum.lock)
    end
    return accum
end

function in_place_add!(accum::Accumulator{<:Real}, increment::Real)
    lock(accum.lock)
    try
        accum.value = accum.value + increment
    finally
        unlock(accum.lock)
    end
    return accum
end

function in_place_add!(accum::Accumulator{<:Array}, increment, scale_factor::Real)
    lock(accum.lock)
    try
        @simd for i in 1:length(accum.value)
            accum.value[i] += increment[i] * scale_factor
        end
    finally
        unlock(accum.lock)
    end
    return accum
end

function in_place_add!(accum::Accumulator{<:Array}, increment)
    lock(accum.lock)
    try
        @simd for i in 1:length(accum.value)
            accum.value[i] += increment[i]
        end
    finally
        unlock(accum.lock)
    end
    return accum
end




###################################
# parameter stores and optimizers #
###################################

"""
    optimizer = init_optimizer(
        conf, parameter_ids::Vector,
        store=default_julia_parameter_store)

Initialize an iterative gradient-based optimizer that mutates a single parameter store.

The first argument defines the mathematical behavior of the update, the second argument defines the set of parameters to which the update should be applied at each iteration, and the third argument gives the location of the parameter values and their gradient accumulators.

Add support for new parameter store types or new optimization configurations by (i) defining a new Julia type for an optimizer that applies the configuration type to the parameter store type, and (ii) implementing this method so that it returns an instance of your new type, and (iii) implement `apply_update!` for your new type.
"""
function init_optimizer(conf, parameter_ids, store=default_julia_parameter_store)
    error("Not implemented")
end

"""
    apply_update!(optimizer)

Apply one iteration of a gradient-based optimization update.

Extend this method to add support for new parameter store types or new optimization configurations.
"""
function apply_update!(optimizer)
    error("Not implemented")
end

struct CompositeOptimizer
    conf::Any
    optimizers::Dict{Any,Any}
    function CompositeOptimizer(conf, parameter_stores_to_ids)
        optimizers = Dict{Any,Any}()
        for (store, parameter_ids) in parameter_stores_to_ids
            optimizers[store] = init_optimizer(conf, collect(parameter_ids), store)
        end
        new(conf, optimizers)
    end
end

"""

    optimizer = init_optimizer(conf, parameter_stores_to_ids::Dict{Any,Vector})

Construct a special type of optimizer that updates parameters in multiple parameter stores.

The second argument defines the set of parameters to which the update should be applied at each iteration,
The parameters are given in a map from parameter store to a vector of IDs of parameters within that parameter store.

NOTE: You do _not_ need to extend this method to extend support for new parameter store types or new optimization configurations.
"""
function init_optimizer(conf, parameter_stores_to_ids::Dict)
    return CompositeOptimizer(conf, parameter_stores_to_ids)
end

"""
    optimizer = init_optimizer(
        conf, gen_fn::GenerativeFunction; parameter_context=default_parameter_context)

Convenience method that constructs an optimizer that updates all parameters used by the given generative function, even when the parameters exist in multiple parameter stores.
"""
function init_optimizer(
        conf, gen_fn::GenerativeFunction; parameter_context=default_parameter_context)
    return CompositeOptimizer(conf, get_parameters(gen_fn, parameter_context))
end


function apply_update!(composite_opt::CompositeOptimizer)
    for opt in values(composite_opt.optimizers)
        apply_update!(opt)
    end
    return nothing
end


#########
# Julia #
#########

struct JuliaParameterStore
    values::Dict{GenerativeFunction,Dict{Symbol,Any}}
    gradient_accumulators::Dict{GenerativeFunction,Dict{Symbol,Accumulator}}
end

"""
    
    store = JuliaParameterStore()

Construct a parameter store stores the state of parameters in the memory of the Julia runtime as Julia values.

There is a global Julia parameter store automatically created and named `Gen.default_julia_parameter_store`.

Gradient accumulation is thread-safe (see [`increment_gradient!`](@ref)).
"""
function JuliaParameterStore()
    return JuliaParameterStore(
        Dict{GenerativeFunction,Dict{Symbol,Any}}(),
        Dict{GenerativeFunction,Dict{Symbol,Accumulator}}())
end

function get_local_parameters(store::JuliaParameterStore, gen_fn)
    if !haskey(store.values, gen_fn)
        return Dict{Symbol,Any}()
    else
        return store.values[gen_fn]
    end
end

# for looking up in a parameter context when tracing (simulate, generate)
# once a trace is generated, it is bound to use a particular store
"""
    JULIA_PARAMETER_STORE_KEY 

If a parameter context contains a value for this key, then the value is a `JuliaParameterStore`.
"""
const JULIA_PARAMETER_STORE_KEY = :julia_parameter_store 

function get_julia_store(context::Dict)
    return context[JULIA_PARAMETER_STORE_KEY]::JuliaParameterStore
end

"""
    default_julia_parameter_store::JuliaParameterStore

The default global Julia parameter store.
"""
const default_julia_parameter_store = JuliaParameterStore()


"""
    init_parameter!(
        id::Tuple{GenerativeFunction,Symbol}, value,
        store::JuliaParameterStore=default_julia_parameter_store)

Initialize the the value of a named trainable parameter of a generative function.

Also initializes the gradient accumulator for that parameter to `zero(value)`.

Example:
```julia
init_parameter!((foo, :theta), 0.6)
```

Not thread-safe.
"""
function init_parameter!(
        id::Tuple{GenerativeFunction,Symbol}, value,
        store::JuliaParameterStore=default_julia_parameter_store)
    (gen_fn, name) = id
    if !haskey(store.values, gen_fn)
        store.values[gen_fn] = Dict{Symbol,Any}()
    end
    store.values[gen_fn][name] = value
    if !haskey(store.gradient_accumulators, gen_fn)
        store.gradient_accumulators[gen_fn] = Dict{Symbol,Any}()
    end
    store.gradient_accumulators[gen_fn][name] = Accumulator(zero(value))
    return nothing
end

"""
    reset_gradient!(
        id::Tuple{GenerativeFunction,Symbol},
        store::JuliaParameterStore=default_julia_parameter_store)

Reset the gradient accumulator for a trainable parameter.

Not thread-safe.
"""
function reset_gradient!(
        id::Tuple{GenerativeFunction,Symbol},
        store::JuliaParameterStore=default_julia_parameter_store)
    (gen_fn, name) = id
    local value::Any
    try
        value = store.values[gen_fn][name]
    catch KeyError
        @error "parameter $name of $gen_fn was not initialized"
        rethrow()
    end
    if !haskey(store.gradient_accumulators, gen_fn)
        store.gradient_accumulators[gen_fn] = Dict{Symbol,Any}()
    end
    if haskey(store.gradient_accumulators[gen_fn], name)
        fill_with_zeros!(store.gradient_accumulators[gen_fn][name])
    else
        store.gradient_accumulators[gen_fn][name] = Accumulator(zero(value))
    end
    return nothing
end

"""
    increment_gradient!(
        id::Tuple{GenerativeFunction,Symbol}, increment, scale_factor::Real,
        store::JuliaParameterStore=default_julia_parameter_store)

Increment the gradient accumulator for a parameter.

The increment is scaled by the given scale_factor.

Thread-safe (multiple threads can increment the gradient of the same parameter concurrently).
"""
function increment_gradient!(
        id::Tuple{GenerativeFunction,Symbol}, increment, scale_factor,
        store::JuliaParameterStore=default_julia_parameter_store)
    (gen_fn, name) = id
    try
        in_place_add!(store.gradient_accumulators[gen_fn][name], increment, scale_factor)
    catch KeyError
        @error "parameter $name of $gen_fn was not initialized"
        rethrow()
    end
    return nothing
end

"""
    increment_gradient!(
        id::Tuple{GenerativeFunction,Symbol}, increment,
        store::JuliaParameterStore=default_julia_parameter_store)

Increment the gradient accumulator for a parameter.

Thread-safe (multiple threads can increment the gradient of the same parameter concurrently).
"""
function increment_gradient!(
        id::Tuple{GenerativeFunction,Symbol}, increment,
        store::JuliaParameterStore=default_julia_parameter_store)
    accumulator = get_gradient_accumulator(id, store)
    in_place_add!(accumulator, increment)
    return nothing
end


"""
    accum::Accumulator = get_gradient_accumulator!(
        id::Tuple{GenerativeFunction,Symbol},
        store::JuliaParameterStore=default_julia_parameter_store)

Return the gradient accumulator for a parameter.

Not thread-safe.
"""
function get_gradient_accumulator(
        id::Tuple{GenerativeFunction,Symbol},
        store::JuliaParameterStore=default_julia_parameter_store)
    (gen_fn, name) = id
    try
        return store.gradient_accumulators[gen_fn][name]
    catch KeyError
        @error "parameter $name of $gen_fn was not initialized"
        rethrow()
    end
end

"""
    value::Union{Real,Array} = get_parameter_value(
        id::Tuple{GenerativeFunction,Symbol},
        store::JuliaParameterStore=default_julia_parameter_store)

Get the current value of a parameter.

Not thread-safe.
"""
function get_parameter_value(
        id::Tuple{GenerativeFunction,Symbol},
        store::JuliaParameterStore=default_julia_parameter_store)
    (gen_fn, name) = id
    try
        return store.values[gen_fn][name]
    catch KeyError
        @error "parameter $name of $gen_fn was not initialized"
        rethrow()
    end
end

"""
    set_parameter_value!(
        id::Tuple{GenerativeFunction,Symbol}, value::Union{Real,Array},
        store::JuliaParameterStore=default_julia_parameter_store)

Set the value of a parameter.

Not thread-safe.
"""
function set_parameter_value!(
        id::Tuple{GenerativeFunction,Symbol}, value::Union{Real,Array},
        store::JuliaParameterStore=default_julia_parameter_store)
    (gen_fn, name) = id
    try
        store.values[gen_fn][name] = value
    catch KeyError
        @error "parameter $name of $gen_fn was not initialized"
        rethrow()
    end
    return nothing
end

"""
    gradient::Union{Real,Array} = get_gradient(
        id::Tuple{GenerativeFunction,Symbol},
        store::JuliaParameterStore=default_julia_parameter_store)

Get the current value of the gradient accumulator for a parameter.

Not thread-safe.
"""
function get_gradient(
        id::Tuple{GenerativeFunction,Symbol},
        store::JuliaParameterStore=default_julia_parameter_store)
    (gen_fn, name) = id
    try
        return get_value(store.gradient_accumulators[gen_fn][name])
    catch KeyError
        @error "parameter $name of $gen_fn was not initialized"
        rethrow()
    end
end

#####################################################
# Optimizer implementations for JuliaParameterStore #
#####################################################

mutable struct FixedStepGradientDescentJulia
    conf::FixedStepGradientDescent
    store::JuliaParameterStore
    parameters::Vector
end

function init_optimizer(
        conf::FixedStepGradientDescent,
        parameters::Vector,
        store::JuliaParameterStore=default_julia_parameter_store)
    return FixedStepGradientDescentJulia(conf, store, parameters)
end

function apply_update!(opt::FixedStepGradientDescentJulia)
    for parameter_id::Tuple{GenerativeFunction,Symbol} in opt.parameters
        value = get_parameter_value(parameter_id, opt.store)
        gradient = get_gradient(parameter_id, opt.store)
        new_value = in_place_add!(value, gradient * opt.conf.step_size)
        set_parameter_value!(parameter_id, new_value, opt.store)
        reset_gradient!(parameter_id, opt.store)
    end
end

mutable struct DecayStepGradientDescentJulia
    conf::DecayStepGradientDescent
    store::JuliaParameterStore
    parameters::Vector
    t::Int
end

function init_optimizer(
        conf::DecayStepGradientDescent,
        parameters::Vector,
        store::JuliaParameterStore=default_julia_parameter_store)
    return DecayStepGradientDescentJulia(conf, store, parameters, 1)
end

function apply_update!(opt::DecayStepGradientDescentJulia)
    step_size_init = opt.conf.step_size_init
    step_size_beta = opt.conf.step_size_beta
    step_size = step_size_init * (step_size_beta + 1) / (step_size_beta + opt.t)
    for parameter_id::Tuple{GenerativeFunction,Symbol} in opt.parameters
        value = get_parameter_value(parameter_id, opt.store)
        gradient = get_gradient(parameter_id, opt.store)
        new_value = in_place_add!(value, gradient * step_size)
        set_parameter_value!(parameter_id, new_value, opt.store)
        reset_gradient!(parameter_id, opt.store)
    end
    opt.t += 1
end

# TODO implement other optimizers (ADAM, etc.)

#############################
# default parameter context #
#############################


"""
    default_parameter_context::Dict

The default global parameter context, which is initialized to contain the mapping:

    JULIA_PARAMETER_STORE_KEY => Gen.default_julia_parameter_store
"""
const default_parameter_context = Dict{Symbol,Any}(
    JULIA_PARAMETER_STORE_KEY => default_julia_parameter_store)

function get_parameters(gen_fn::GenerativeFunction)
    return get_parameters(gen_fn, default_parameter_context)
end

function simulate(gen_fn::GenerativeFunction, args::Tuple)
    return simulate(gen_fn, args, default_parameter_context)
end

function generate(gen_fn::GenerativeFunction, args::Tuple, choices::ChoiceMap)
    return generate(gen_fn, args, choices, default_parameter_context)
end

function generate(gen_fn::GenerativeFunction, args::Tuple)
    return generate(gen_fn, args, EmptyChoiceMap(), default_parameter_context)
end

function assess(gen_fn::GenerativeFunction, args::Tuple, choices::ChoiceMap)
    return assess(gen_fn, args, choices, default_parameter_context)
end

propose(gen_fn::GenerativeFunction, args::Tuple) = propose(gen_fn, args, default_parameter_context)
