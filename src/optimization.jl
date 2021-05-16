import Parameters

# TODO notes
#
# we should modify the semantics of the log probability contribution to the gradient
# so that everything is gradient descent instead of ascent. this will also fix
# the misnomer names
#
# combinators (map etc.) and call_at! and choice_at! all need to implement get_parameters..
# 
# make changes to src/static_ir/backprop.jl
#
# make changes to src/dynamic/dynamic.jl (use the JuliaParameterStore)
#
# TODO GF untraced needs to reference a parameter store
#
# make changes to src/dynamic/backprop.jl

export in_place_add!

export FixedStepGradientDescent
export DecayStepGradientDescent
export make_optimizer
export apply_update!

export ParameterStore
export JuliaParameterStore
export JuliaParameterID

export initialize_parameter!
export increment_gradient!
export reset_gradient!
export get_parameter_value

#################
# in_place_add! #
#################

function in_place_add! end

function in_place_add!(value::Array, increment)
    @simd for i in 1:length(param)
        value[i] += increment[i]
    end
    return value
end

# this exists so user can use the same function on scalars and arrays
function in_place_add!(param::Real, increment::Real)
    return param + increment
end

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


# TODO add ADAM update

###########################
# thread-safe accumulator # 
###########################

struct Accumulator{T<:Union{Real,Array}}
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

function fill_with_zeros!(accum::Accumulator{Array{T}}) where {T}
    lock(accum.lock)
    try
        fill!(zero(T), accum.arr)
    finally
        unlock(accum.lock)
    end
    return accum
end

function in_place_add!(accum::ThreadsafeAccumulator{Real}, increment::Real, scale_factor::Real)
    lock(accum.lock)
    try
        accum.value = accum.value + increment * scale_factor
    finally
        unlock(accum.lock)
    end
    return accum
end

function in_place_add!(accum::ThreadsafeAccumulator{Real}, increment::Real)
    lock(accum.lock)
    try
        accum.value = accum.value + increment
    finally
        unlock(accum.lock)
    end
    return accum
end

function in_place_add!(accum::ThreadsafeAccumulator{<:Array}, increment, scale_factor::Real)
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

function in_place_add!(accum::ThreadsafeAccumulator{<:Array}, increment)
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




#################################
# ParameterStore and optimizers #
#################################

abstract type ParameterStore end

# TODO docstring, returns an optimizer that has an apply_update! method
function make_optimizer(conf, store::ParameterStore, parameter_ids) end

# TODO docstring
function apply_update!(optimizer) end

struct CompositeOptimizer
    conf::Any
    optimizers::Dict{ParameterStore,Any}
    function CompositeOptimizer(conf, parameters::Dict{ParameterStore,Vector})
        optimizers = Dict{ParameterStore,Any}()
        for (store, parameter_ids) in parameters
            optimizers[store] = make_optimizer(conf, store, parameter_ids)
        end
        new(states, conf)
    end
end

function CompositeOptimizer(conf, gen_fn::GenerativeFunction)
    return CompositeOptimizer(conf, get_parameters(gen_fn))
end

"""
    apply_update!(update::ParamUpdate)

Perform one step of the update.
"""
function apply_update!(composite_opt::CompositeOptimizer)
    for opt in values(composite_opt.optimizers)
        apply_update!(opt)
    end
    return nothing
end


#########
# Julia #
#########

const JuliaParameterID = Tuple{GenerativeFunction,Symbol}

# TODO document
struct JuliaParameterStore
    values::Dict{GenerativeFunction,Dict{Symbol,Any}}
    gradient_accumulators::Dict{GenerativeFunction,Dict{Symbol,GradientAccumulator}}
end

function JuliaParameterStore()
    return JuliaParameterStore(
        Dict{GenerativeFunction,Dict{Symbol,Any}}(),
        Dict{GenerativeFunction,Dict{Symbol,GradientAccumulator}}())
end

get_local_parameters(store::JuliaParameterStore, gen_fn) = store.values[gen_fn]

# TODO document
const default_parameter_context = Dict{Symbol,Any}()
const default_julia_parameter_store = JuliaParameterStore()

# for looking up in a parameter context when tracing (simulate, generate)
# TODO make the parametr context another argument to simulate and generate
# once a trace is generated, it is bound to use a particular store
const JULIA_PARAMETER_STORE_KEY = :julia_parameter_store 

function get_julia_store(context::Dict{Symbol,Any})
    if haskey(context, JULIA_PARAMETER_STORE_KEY)
        return context[JULIA_PARAMETER_STORE_KEY]
    else
        return default_julia_parameter_store
    end
end

"""
    initialize_parameter!(store::JuliaParameterStore, id::JuliaParameterID, value)

Initialize the the value of a named trainable parameter of a generative function.

Also generates the gradient accumulator for that parameter to `zero(value)`.

Example:
```julia
initialize_parameter!(foo, :theta, 0.6)
```

Not thread-safe.
"""
function initialize_parameter!(store::JuliaParameterStore, id::JuliaParameterID, value)
    (gen_fn, name) = id
    if !haskey(store.values, gen_fn)
        store.values[gen_fn] = Dict{Symbol,Any}()
    end
    store.values[gen_fn][name] = value
    reset_gradient!(store, id)
    return nothing
end

# TODO docstring (not thread-safe)
function reset_gradient!(store::JuliaParameterStore, id::JuliaParameterID)
    (gen_fn, name) = id
    try
        value = store.values[gen_fn][name]
    catch KeyError
        @error "parameter not initialized: $id"
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

# TODO docstring (thread-safe)
function increment_gradient!(
        store::JuliaParameterStore, id::JuliaParameterID,
        increment, scale_factor)
    (gen_fn, name) = id
    try
        in_place_add!(store.gradient_accumulators[gen_fn][name], increment, scale_factor)
    catch KeyError
        @error "parameter not initialized: $id"
        rethrow()
    end
    return nothing
end

# TODO docstring (thread-safe)
function increment_gradient!(
        store::JuliaParameterStore, id::JuliaParameterID,
        increment)
    (gen_fn, name) = id
    try
        in_place_add!(store.gradient_accumulators[gen_fn][name], increment)
    catch KeyError
        @error "parameter not initialized: $id"
        rethrow()
    end
    return nothing
end

# TODO docstring (not thread-safe)
function get_parameter_value(store::JuliaParameterStore, id::JuliaParameterID)
    (gen_fn, name) = id
    try
        return state.values[gen_fn][name]
    catch KeyError
        @error "parameter not initialized: $id"
        rethrow()
    end
end

# TODO docstring (not thread-safe)
function set_parameter_value!(store::JuliaParameterStore, id::JuliaParameterID, value)
    (gen_fn, name) = id
    try
        store.values[gen_fn][name] = value
    catch KeyError
        @error "parameter not initialized: $id"
        rethrow()
    end
    return nothing
end

# TODO docstring (not thread-safe)
function get_gradient(store::JuliaParameterStore, id::JuliaParameterID)
    (gen_fn, name) = id
    try
        return get_value(store.gradient_accumulators[gen_fn][name])
    catch KeyError
        @error "parameter not initialized: $id"
        rethrow()
    end
end

#####################################################
# Optimizer implementations for JuliaParameterStore #
#####################################################

mutable struct FixedStepGradientDescentJulia
    conf::FixedStepGradientDescent
    store::JuliaParameterStore
    parameters::Vector{JuliaParameterID}
end

function make_optimizer(
        conf::FixedStepGradientDescent,
        store::JuliaParameterStore,
        parameters::Vector{JuliaParameterID})
    return FixedStepGradientDescentJulia(conf, store, parameters)
end

# TODO docstring (not thread-safe)
function apply_update!(opt::FixedStepGradientDescentJulia)
    for parameter_id in opt.parameters
        value = get_parameter_value(opt.store, parameter_id)
        gradient = get_gradient(opt.store, id)
        new_value = in_place_add!(value, gradient * opt.conf.step_size)
        set_parameter_value!(store, parameter_id, new_value)
        reset_gradient!(store, parameter_id)
    end
end

# TODO implement other optimizers
