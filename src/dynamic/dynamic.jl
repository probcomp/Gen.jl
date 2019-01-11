include("trace.jl")

struct DynamicDSLFunction{T} <: GenerativeFunction{T,DynamicDSLTrace}
    params_grad::Dict{Symbol,Any}
    params::Dict{Symbol,Any}
    arg_types::Vector{Type}
    julia_function::Function
    update_julia_function::Function
    has_argument_grads::Vector{Bool}
    accepts_output_grad::Bool
end

function DynamicDSLFunction(arg_types::Vector{Type},
                     julia_function::Function,
                     update_julia_function::Function,
                     has_argument_grads, ::Type{T},
                     accepts_output_grad::Bool) where {T}
    params_grad = Dict{Symbol,Any}()
    params = Dict{Symbol,Any}()
    DynamicDSLFunction{T}(params_grad, params, arg_types,
                julia_function, update_julia_function,
                has_argument_grads, accepts_output_grad)
end

accepts_output_grad(gen_fn::DynamicDSLFunction) = gen_fn.accepts_output_grad

function (g::DynamicDSLFunction)(args...)
    (trace, _) = initialize(g, args, EmptyAssignment())
    get_retval(trace)
end

function exec(gf::DynamicDSLFunction, state, args::Tuple)
    gf.julia_function(state, args...)
end

function exec_for_update(gf::DynamicDSLFunction, state, args::Tuple)
    gf.update_julia_function(state, args...)
end

# whether there is a gradient of score with respect to each argument
# it returns 'nothing' for those arguemnts that don't have a derivatice
has_argument_grads(gen::DynamicDSLFunction) = gen.has_argument_grads

macro addr(expr::Expr, addr, addrdiff)
    if expr.head != :call
        error("syntax error in $DYNAMIC_DSL_ADDR at $(expr)")
    end
    fn = esc(expr.args[1])
    args = map(esc, expr.args[2:end])
    Expr(:call, :addr, esc(state), fn, Expr(:tuple, args...), esc(addr), esc(addrdiff))
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


#####################
# static parameters #
#####################

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

function set_param!(gf::DynamicDSLFunction, name::Symbol, value)
    gf.params[name] = value
end

function get_param(gf::DynamicDSLFunction, name::Symbol)
    gf.params[name]
end

function get_param_grad(gf::DynamicDSLFunction, name::Symbol)
    gf.params_grad[name]
end

function zero_param_grad!(gf::DynamicDSLFunction, name::Symbol)
    gf.params_grad[name] = zero(gf.params[name])
end

"""
    init_param!(gen_fn, name::Symbol, value)

Initialize the the value of a named static parameter of a generative function.

Also initializes the gradient accumulator for that parameter to `zero(value)`.
"""
function init_param!(gf::DynamicDSLFunction, name::Symbol, value)
    set_param!(gf, name, value)
    zero_param_grad!(gf, name)
end

get_param_names(gf::DynamicDSLFunction) = keys(gf.params)

#########
# diffs #
#########

macro argdiff()
    Expr(:call, :get_arg_diff, esc(state))
end

macro choicediff(addr)
    Expr(:call, :get_choice_diff, esc(state), esc(addr))
end

macro calldiff(addr)
    Expr(:call, :get_call_diff, esc(state), esc(addr))
end

macro retdiff(value)
    Expr(:call, :set_ret_diff!, esc(state), esc(value))
end

set_ret_diff!(state, value) = state.retdiff = value

get_arg_diff(state) = state.argdiff


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

function all_visited(visited::AddressSet, assmt::Assignment)
    allvisited = true
    for (key, _) in get_values_shallow(assmt)
        allvisited = allvisited && has_leaf_node(visited, key)
    end
    for (key, subassmt) in get_subassmts_shallow(assmt)
        if !has_leaf_node(visited, key)
            if has_internal_node(visited, key)
                subvisited = get_internal_node(visited, key)
            else
                subvisited = EmptyAddressSet()
            end
            allvisited = allvisited && all_visited(subvisited, subassmt)
        end
    end
    allvisited
end

function get_unvisited(visited::AddressSet, assmt::Assignment)
    unvisited = DynamicAssignment()
    for (key, _) in get_values_shallow(assmt)
        if !has_leaf_node(visited, key)
            set_value!(unvisited, key, get_value(assmt, key))
        end
    end
    for (key, subassmt) in get_subassmts_shallow(assmt)
        if !has_leaf_node(visited, key)
            if has_internal_node(visited, key)
                subvisited = get_internal_node(visited, key)
            else
                subvisited = EmptyAddressSet()
            end
            sub_unvisited = get_unvisited(subvisited, subassmt)
            set_subassmt!(unvisited, key, sub_unvisited)
        end
    end
    unvisited
end

get_visited(visitor) = visitor.visited

function check_no_subassmt(constraints::Assignment, addr)
    if !isempty(get_subassmt(constraints, addr))
        error("Expected a value at address $addr but found a sub-assignment")
    end
end

function check_no_value(constraints::Assignment, addr)
    if has_value(constraints, addr)
        error("Expected a sub-assignment at address $addr but found a value")
    end
end

function lightweight_retchange_already_set_err()
    error("@retdiff! was already used")
end

function gen_fn_changed_error(addr)
    error("Generative function changed at address: $addr")
end

include("diff.jl")
include("initialize.jl")
include("propose.jl")
include("assess.jl")
include("project.jl")
include("force_update.jl")
include("fix_update.jl")
include("free_update.jl")
include("extend.jl")
include("backprop.jl")

export DynamicDSLFunction
export set_param!, get_param, get_param_grad, zero_param_grad!, init_param!, get_param_names
export @param
export @addr
export @gen
export @choicediff
export @calldiff
export @argdiff
export @retdiff
