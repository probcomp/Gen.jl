export register_parameters!

include("trace.jl")

"""
    DynamicDSLFunction{T} <: GenerativeFunction{T,DynamicDSLTrace}

A generative function based on a shallowly embedding modeling language based on Julia functions.

Constructed using the `@gen` keyword.
Most methods in the generative function interface involve a end-to-end execution of the function.
"""
mutable struct DynamicDSLFunction{T} <: GenerativeFunction{T,DynamicDSLTrace}
    arg_types::Vector{Type}
    has_defaults::Bool
    arg_defaults::Vector{Union{Some{Any},Nothing}}
    julia_function::Function
    has_argument_grads::Vector{Bool}
    accepts_output_grad::Bool
    parameters::Union{Vector,Function}
end

function DynamicDSLFunction(arg_types::Vector{Type},
                     arg_defaults::Vector{Union{Some{Any},Nothing}},
                     julia_function::Function,
                     has_argument_grads, ::Type{T},
                     accepts_output_grad::Bool) where {T}
    has_defaults = any(arg -> arg != nothing, arg_defaults)
    return DynamicDSLFunction{T}(arg_types,
                has_defaults, arg_defaults,
                julia_function,
                has_argument_grads, accepts_output_grad, [])
end

function get_parameters(gen_fn::DynamicDSLFunction, parameter_context)
    if isa(gen_fn.parameters, Vector)
        julia_store = get_julia_store(parameter_context)
        parameter_stores_to_ids = Dict{Any,Vector}()
        parameter_ids = Tuple{GenerativeFunction,Symbol}[]
        for param in gen_fn.parameters
            if isa(param, Tuple{GenerativeFunction,Symbol})
                push!(parameter_ids, param)
            elseif isa(param, Symbol)
                push!(parameter_ids, (gen_fn, param))
            else
                throw(ArgumentError("Invalid parameter declaration for DML generative function $gen_fn: $param"))
            end
        end
        parameter_stores_to_ids[julia_store] = parameter_ids
        return parameter_stores_to_ids
    elseif isa(gen_fn.parameters, Function)
        return gen_fn.parameters(parameter_context)
    end
end

"""
    register_parameters!(gen_fn::DynamicDSLFunction, parameters)

Register the altrainable parameters that are used by a DML generative function.

This includes all parameters used within any calls made by the generative function.

There are two variants:

# TODO document the variants
"""
function register_parameters!(gen_fn::DynamicDSLFunction, parameters)
    gen_fn.parameters = parameters
    return nothing
end

function Base.show(io::IO, gen_fn::DynamicDSLFunction)
    print(io, "Gen DML generative function: $(gen_fn.julia_function)")
end

function Base.show(io::IO, ::MIME"text/plain", gen_fn::DynamicDSLFunction)
    print(io, "Gen DML generative function: $(gen_fn.julia_function)")
end

function DynamicDSLTrace(
        gen_fn::T, args, parameter_store::JuliaParameterStore,
        parameter_context, registered_julia_parameters) where {T<:DynamicDSLFunction}
    # pad args with default values, if available
    if gen_fn.has_defaults && length(args) < length(gen_fn.arg_defaults)
        defaults = gen_fn.arg_defaults[length(args)+1:end]
        defaults = map(x -> something(x), defaults)
        args = Tuple(vcat(collect(args), defaults))
    end
    return DynamicDSLTrace{T}(
        gen_fn, args, parameter_store, parameter_context, registered_julia_parameters)
end

accepts_output_grad(gen_fn::DynamicDSLFunction) = gen_fn.accepts_output_grad

mutable struct GFUntracedState
    gen_fn::GenerativeFunction
    parameter_store::JuliaParameterStore
end

get_parameter_store(state::GFUntracedState) = state.parameter_store
get_parameter_id(state::GFUntracedState, name::Symbol) = (state.gen_fn, name)

function (gen_fn::DynamicDSLFunction)(args...)
    state = GFUntracedState(gen_fn, default_julia_parameter_store)
    gen_fn.julia_function(state, args...)
end

function exec(gen_fn::DynamicDSLFunction, state, args::Tuple)
    gen_fn.julia_function(state, args...)
end

function splice(state, gen_fn::DynamicDSLFunction, args::Tuple)
    prev_gen_fn = get_active_gen_fn(state)
    state.active_gen_fn = gen_fn
    retval = exec(gen_fn, state, args)
    set_active_gen_fn!(state, prev_gen_fn)
    return retval
end

# whether there is a gradient of score with respect to each argument
# it returns 'nothing' for those arguemnts that don't have a derivatice
has_argument_grads(gen::DynamicDSLFunction) = gen.has_argument_grads

"Global reference to the GFI state for the dynamic modeling language."
const state = gensym("state")

"Implementation of @trace for the dynamic modeling language."
function dynamic_trace_impl(expr::Expr)
    @assert expr.head == :gentrace "Not a Gen trace expression."
    call, addr = expr.args[1], expr.args[2]
    if (call.head != :call) error("syntax error in @trace at $(call)") end
    fn = call.args[1]
    args = Expr(:tuple, call.args[2:end]...)
    if addr != nothing
        addr = something(addr)
        return Expr(:call, GlobalRef(@__MODULE__, :traceat), state, fn, args, addr)
    else
        return Expr(:call, GlobalRef(@__MODULE__, :splice), state, fn, args)
    end
end

# Defaults for untraced execution
@inline traceat(state::GFUntracedState, gen_fn::GenerativeFunction, args, key) =
    gen_fn(args...)

@inline traceat(state::GFUntracedState, dist::Distribution, args, key) =
    random(dist, args...)

@inline splice(state::GFUntracedState, gen_fn::DynamicDSLFunction, args::Tuple) =
    gen_fn(args...)

########################
# trainable parameters #
########################

"Implementation of @param for the dynamic modeling language."
function dynamic_param_impl(expr::Expr)
    @assert expr.head == :genparam "Not a Gen param expression."
    name = expr.args[1]
    Expr(:(=), name, Expr(:call, GlobalRef(@__MODULE__, :read_param!), state, QuoteNode(name)))
end

function read_param!(state, name::Symbol)
    parameter_id = get_parameter_id(state, name)
    store = get_parameter_store(state)
    return get_parameter_value(parameter_id, store)
end

##################
# AddressVisitor #
##################

struct AddressVisitor
    visited::DynamicSelection
end

AddressVisitor() = AddressVisitor(DynamicSelection())

function visit!(visitor::AddressVisitor, addr)
    if addr in visitor.visited
        error("Attempted to visit address $addr, but it was already visited")
    end
    push!(visitor.visited, addr)
end

function all_visited(visited::Selection, choices::ChoiceMap)
    allvisited = true
    for (key, _) in get_values_shallow(choices)
        allvisited = allvisited && (key in visited)
    end
    for (key, submap) in get_submaps_shallow(choices)
        if !(key in visited)
            subvisited = visited[key]
            allvisited = allvisited && all_visited(subvisited, submap)
        end
    end
    allvisited
end

function get_unvisited(visited::Selection, choices::ChoiceMap)
    unvisited = choicemap()
    for (key, _) in get_values_shallow(choices)
        if !(key in visited)
            set_value!(unvisited, key, get_value(choices, key))
        end
    end
    for (key, submap) in get_submaps_shallow(choices)
        if !(key in visited)
            subvisited = visited[key]
            sub_unvisited = get_unvisited(subvisited, submap)
            set_submap!(unvisited, key, sub_unvisited)
        end
    end
    unvisited
end

get_visited(visitor) = visitor.visited

function check_no_submap(constraints::ChoiceMap, addr)
    if !isempty(get_submap(constraints, addr))
        error("Expected a value at address $addr but found a sub-assignment")
    end
end

function check_no_value(constraints::ChoiceMap, addr)
    if has_value(constraints, addr)
        error("Expected a sub-assignment at address $addr but found a value")
    end
end

function gen_fn_changed_error(addr)
    error("Generative function changed at address: $addr")
end

include("simulate.jl")
include("generate.jl")
include("propose.jl")
include("assess.jl")
include("project.jl")
include("update.jl")
include("regenerate.jl")
include("backprop.jl")

export DynamicDSLFunction
