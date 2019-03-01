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
    julia_function::Function
    has_argument_grads::Vector{Bool}
    accepts_output_grad::Bool
end

function DynamicDSLFunction(arg_types::Vector{Type},
                     julia_function::Function,
                     has_argument_grads, ::Type{T},
                     accepts_output_grad::Bool) where {T}
    params_grad = Dict{Symbol,Any}()
    params = Dict{Symbol,Any}()
    DynamicDSLFunction{T}(params_grad, params, arg_types,
                julia_function,
                has_argument_grads, accepts_output_grad)
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
        error("Syntax in error in @static at $(expr_or_symbol)")
    end
    Expr(:(=), esc(name), Expr(:call, :read_param, esc(state), QuoteNode(name)))
end

##################
# AddressVisitor #
##################

struct AddressVisitor
    visited::DynamicAddressSet
end

AddressVisitor() = AddressVisitor(DynamicAddressSet())

function visit!(visitor::AddressVisitor, addr)
    push!(visitor.visited, addr)
end

function all_visited(visited::AddressSet, choices::ChoiceMap)
    allvisited = true
    for (key, _) in get_values_shallow(choices)
        allvisited = allvisited && has_leaf_node(visited, key)
    end
    for (key, submap) in get_submaps_shallow(choices)
        if !has_leaf_node(visited, key)
            if has_internal_node(visited, key)
                subvisited = get_internal_node(visited, key)
            else
                subvisited = EmptyAddressSet()
            end
            allvisited = allvisited && all_visited(subvisited, submap)
        end
    end
    allvisited
end

function get_unvisited(visited::AddressSet, choices::ChoiceMap)
    unvisited = choicemap()
    for (key, _) in get_values_shallow(choices)
        if !has_leaf_node(visited, key)
            set_value!(unvisited, key, get_value(choices, key))
        end
    end
    for (key, submap) in get_submaps_shallow(choices)
        if !has_leaf_node(visited, key)
            if has_internal_node(visited, key)
                subvisited = get_internal_node(visited, key)
            else
                subvisited = EmptyAddressSet()
            end
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
include("extend.jl")
include("backprop.jl")

export DynamicDSLFunction
export @param
export @trace
export @gen
