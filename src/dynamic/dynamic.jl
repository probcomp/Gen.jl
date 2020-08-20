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

########################
# trainable parameters #
########################

"Implementation of @param for the dynamic modeling language."
function dynamic_param_impl(expr::Expr)
    @assert expr.head == :genparam "Not a Gen param expression."
    name = expr.args[1]
    Expr(:(=), name, Expr(:call, GlobalRef(@__MODULE__, :read_param), state, QuoteNode(name)))
end

function read_param(state, name::Symbol)
    if haskey(state.params, name)
        state.params[name]
    else
        throw(UndefVarError(name))
    end
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

all_constraints_visited(::Selection, ::Value) = false
all_constraints_visited(::AllSelection, ::Value) = true
all_constraints_visited(::Selection, ::Selection) = true # we're allowed to not visit selections
all_constraints_visited(::Selection, ::EmptyAddressTree) = true
all_constraints_visited(::AllSelection, ::EmptyAddressTree) = true
function all_constraints_visited(visited::Selection, spec::UpdateSpec)
    for (key, subtree) in get_subtrees_shallow(spec)
        if !all_constraints_visited(get_subselection(visited, key), subtree)
            return false
        end
    end
    return true
end

get_unvisited(::Selection, v::Value) = v
get_unvisited(::AllSelection, v::Value) = EmptyChoiceMap()
function get_unvisited(visited::Selection, choices::ChoiceMap)
    unvisited = choicemap()
    for (key, submap) in get_submaps_shallow(choices)
        sub_unvisited = get_unvisited(get_subselection(visited, key), submap)
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
include("backprop.jl")

export DynamicDSLFunction
