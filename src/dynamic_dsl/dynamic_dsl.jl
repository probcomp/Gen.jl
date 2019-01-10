include("trace.jl")

const DYNAMIC_DSL_ADDR = Symbol("@addr")
const DYNAMIC_DSL_DIFF = Symbol("@diff")
const DSL_STATIC_ANNOTATION = :static
const DSL_ARG_GRAD_ANNOTATION = :grad
const DSL_RET_GRAD_ANNOTATION = :grad

struct Argument
    name::Symbol
    typ::Union{Symbol,Expr}
    annotations::Set{Symbol}
end

Argument(name, typ) = Argument(name, typ, Set{Symbol}())


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

function parse_annotations(annotations_expr)
    annotations = Set{Symbol}()
    if isa(annotations_expr, Symbol)
        push!(annotations, annotations_expr)
    elseif isa(annotations_expr, Expr) && annotations_expr.head == :tuple
        for annotation in annotations_expr.args
            if !isa(annotation, Symbol)
                error("syntax error in annotations_expr at $annotation")
            else
                push!(annotations, annotation)
            end
        end
    else
        error("syntax error in annotations at $annotations")
    end
    annotations
end

macro gen(annotations_expr, ast)

    # parse the annotations
    annotations = parse_annotations(annotations_expr)

    # parse the function definition
    parse_gen_function(ast, annotations)
end

macro gen(ast)
    parse_gen_function(ast, Set{Symbol}())
end

transform_body_for_non_update(ast) = ast

function transform_body_for_non_update(ast::Expr)
    @assert !(ast.head == :macrocall && ast.args[1] == DYNAMIC_DSL_DIFF)

    # remove the argdiff argument from @addr expressions
    if ast.head == :macrocall && ast.args[1] == DYNAMIC_DSL_ADDR
        if length(ast.args) == 5
            @assert isa(ast.args[2], LineNumberNode)
            # remove the last argument
            ast = Expr(ast.head, ast.args[1:4]...)
        end
    end

    # remove any @diff sub-expressions, and recurse
    new_args = Vector()
    for arg in ast.args
        if !(isa(arg, Expr) && arg.head == :macrocall && arg.args[1] == DYNAMIC_DSL_DIFF)
            push!(new_args, transform_body_for_non_update(arg))
        end
    end
    Expr(ast.head, new_args...)
end

transform_body_for_update(ast, in_diff) = ast

function transform_body_for_update(ast::Expr, in_diff::Bool)
    @assert !(ast.head == :macrocall && ast.args[1] == DYNAMIC_DSL_DIFF)
    new_args = Vector()
    for arg in ast.args
        if isa(arg, Expr) && arg.head == :macrocall && arg.args[1] == DYNAMIC_DSL_DIFF
            if in_diff
                error("Got nested @diff at: $arg")
            end
            if length(arg.args) == 2
                push!(new_args, transform_body_for_update(arg.args[2], true))
            elseif length(arg.args) == 3 && isa(arg.args[2], LineNumberNode)
                push!(new_args, transform_body_for_update(arg.args[3], true))
            else
                error("@diff must have just one sub-expression, got $arg")
            end
        else
            push!(new_args, arg)
        end
    end
    Expr(ast.head, new_args...)
end

function parse_arg(expr)
    if isa(expr, Symbol)
        # x
        arg = Argument(expr, :Any)
    elseif isa(expr, Expr) && expr.head == :(::)
        # x::Int
        arg = Argument(expr.args[1], expr.args[2])
    elseif isa(expr, Expr) && expr.head == :call
        # (grad,foo)(x::Int)
        annotations_expr = expr.args[1]
        sub_arg = parse_arg(expr.args[2])
        annotations = parse_annotations(annotations_expr)
        arg = Argument(sub_arg.name, sub_arg.typ, annotations)
    else
        dump(expr)
        error("syntax error in gen function argument at $expr")
    end
    arg
end

function parse_gen_function(ast, annotations)
    if ast.head != :function
        error("syntax error at $ast in $(ast.head)")
    end
    if length(ast.args) != 2
        error("syntax error at $ast in $(ast.args)")
    end
    signature = ast.args[1]
    body = ast.args[2]
    if signature.head == :(::)
        (call_signature, return_type) = signature.args
    elseif signature.head == :call
        (call_signature, return_type) = (signature, :Any)
    else
        error("syntax error at $(signature)")
    end
    name = call_signature.args[1]
    args = map(parse_arg, call_signature.args[2:end])
    static = DSL_STATIC_ANNOTATION in annotations
    if static
        # TODO implement, invoke static_dsl/static_dsl.jl
        #make_static_gen_function(name, args, body, return_type, annotations)
    else
        make_dynamic_gen_function(name, args, body, return_type, annotations)
    end
end

function make_dynamic_gen_function(name, args, body, return_type, annotations)
    # julia function definition (for a non-update behavior)
    escaped_args = map((arg) -> esc(arg.name), args)
    gf_args = [esc(state), escaped_args...]
    julia_fn_defn = Expr(:function,
        Expr(:tuple, gf_args...),
        esc(transform_body_for_non_update(body)))

    # julia function definition for update function
    update_julia_fn_defn = Expr(:function,
        Expr(:tuple, gf_args...),
        esc(transform_body_for_update(body, false)))

    # create generator and assign it a name
    arg_types = map((arg) -> esc(arg.typ), args)
    has_argument_grads = map(
        (arg) -> (DSL_ARG_GRAD_ANNOTATION in arg.annotations), args)
    accepts_output_grad = DSL_RET_GRAD_ANNOTATION in annotations
    Expr(:block,
        Expr(:(=), 
            esc(name),
            Expr(:call, :DynamicDSLFunction,
                quote Type[$(arg_types...)] end,
                julia_fn_defn, update_julia_fn_defn,
                has_argument_grads, return_type,
                accepts_output_grad)))
end

function address_not_found_error_msg(addr)
    "Address $addr not found"
end

function read_param(state, name::Symbol)
    state.params[name]
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
