include("trace.jl")

"""
    DynamicDSLFunction{T} <: GenerativeFunction{T,DynamicDSLTrace}

A generative function based on a shallowly embedding modeling language based on Julia functions.

Constructed using the `@gen` keyword.
Most methods in the generative function interface involve a end-to-end execution of the function.
"""
struct DynamicDSLFunction{T} <: GenerativeFunction{T,DynamicDSLTrace}
    params_grad::Dict{Symbol,Any}
    params::Dict{Symbol,Any}
    arg_types::Vector{Type}
    has_defaults::Bool
    arg_defaults::Vector{Union{Some{Any},Nothing}}
    julia_function::Function
    has_argument_grads::Vector{Bool}
    accepts_output_grad::Bool
end

function DynamicDSLFunction(arg_types::Vector{Type},
                     arg_defaults::Vector{Union{Some{Any},Nothing}},
                     julia_function::Function,
                     has_argument_grads, ::Type{T},
                     accepts_output_grad::Bool) where {T}
    params_grad = Dict{Symbol,Any}()
    params = Dict{Symbol,Any}()
    has_defaults = any(arg -> arg != nothing, arg_defaults)
    DynamicDSLFunction{T}(params_grad, params, arg_types,
                has_defaults, arg_defaults,
                julia_function,
                has_argument_grads, accepts_output_grad)
end

function DynamicDSLTrace(gen_fn::T, args) where {T<:DynamicDSLFunction}
    # pad args with default values, if available
    if gen_fn.has_defaults && length(args) < length(gen_fn.arg_defaults)
        defaults = gen_fn.arg_defaults[length(args)+1:end]
        defaults = map(x -> something(x), defaults)
        args = Tuple(vcat(collect(args), defaults))
    end
    DynamicDSLTrace{T}(gen_fn, args)
end

accepts_output_grad(gen_fn::DynamicDSLFunction) = gen_fn.accepts_output_grad

function (gen_fn::DynamicDSLFunction)(args...)
    (_, _, retval) = propose(gen_fn, args)
    retval
end

function exec(gf::DynamicDSLFunction, state, args::Tuple)
    gf.julia_function(state, args...)
end

# whether there is a gradient of score with respect to each argument
# it returns 'nothing' for those arguemnts that don't have a derivatice
has_argument_grads(gen::DynamicDSLFunction) = gen.has_argument_grads

const state = gensym("state")

macro trace(expr::Expr, addr)
    if expr.head != :call
        error("syntax error in @trace at $(expr)")
    end
    fn = esc(expr.args[1])
    args = map(esc, expr.args[2:end])
    Expr(:call, :traceat, esc(state), fn, Expr(:tuple, args...), esc(addr))
end

macro trace(expr::Expr)
    if expr.head != :call
        error("syntax error in @trace at $(expr)")
    end
    invocation = expr.args[1]
    args = esc(Expr(:tuple, expr.args[2:end]...))
    Expr(:call, :splice, esc(state), esc(invocation), args)
end

function address_not_found_error_msg(addr)
    "Address $addr not found"
end

function read_param(state, name::Symbol)
    if haskey(state.params, name)
        state.params[name]
    else
        throw(UndefVarError(name))
    end
end


########################
# trainable parameters #
########################

macro param(expr_or_symbol)
    local name::Symbol
    if isa(expr_or_symbol, Symbol)
        name = expr_or_symbol
    elseif expr_or_symbol.head == :(::)
        name = expr_or_symbol.args[1]
    else
        error("Syntax in error in @param at $(expr_or_symbol)")
    end
    Expr(:(=), esc(name), Expr(:call, :read_param, esc(state), QuoteNode(name)))
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

all_visited(::Selection, ::ValueChoiceMap) = false
all_visited(::AllSelection, ::ValueChoiceMap) = true
function all_visited(visited::Selection, choices::ChoiceMap)
    for (key, submap) in get_submaps_shallow(choices)
        if !all_visited(visited[key], submap)
            return false
        end
    end
    return true
end

get_unvisited(::Selection, v::ValueChoiceMap) = v
get_unvisited(::AllSelection, v::ValueChoiceMap) = EmptyChoiceMap()
function get_unvisited(visited::Selection, choices::ChoiceMap)
    unvisited = choicemap()
    for (key, submap) in get_submaps_shallow(choices)
        sub_unvisited = get_unvisited(visited[key], submap)
        set_submap!(unvisited, key, sub_unvisited)
    end
    unvisited
end

get_visited(visitor) = visitor.visited

function check_is_empty(constraints::ChoiceMap, addr)
    if !isempty(get_submap(constraints, addr))
        error("Expected a value or EmptyChoiceMap at address $addr but found a sub-assignment")
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
export @param
export @trace
export @gen
