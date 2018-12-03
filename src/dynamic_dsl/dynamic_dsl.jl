include("trace.jl")

struct DynamicDSLFunction <: GenerativeFunction{Any,GFTrace}
    params_grad::Dict{Symbol,Any}
    params::Dict{Symbol,Any}
    arg_types::Vector{Type}
    julia_function::Function
    update_julia_function::Function
    has_argument_grads::Vector{Bool}
    accepts_output_grad::Bool
end

function DynamicDSLFunction(arg_types::Vector{Type}, julia_function::Function,
                     update_julia_function::Function,
                     has_argument_grads, accepts_output_grad::Bool)
    params_grad = Dict{Symbol,Any}()
    params = Dict{Symbol,Any}()
    DynamicDSLFunction(params_grad, params, arg_types,
                julia_function, update_julia_function,
                has_argument_grads, accepts_output_grad)
end

function (g::DynamicDSLFunction)(args...)
    trace = simulate(g, args)
    call = get_call_record(trace)
    call.retval
end


exec(gf::DynamicDSLFunction, state, args::Tuple) = gf.julia_function(state, args...)
exec_for_update(gf::DynamicDSLFunction, state, args::Tuple) = gf.update_julia_function(state, args...)

get_static_argument_types(gen::DynamicDSLFunction) = gen.arg_types

# whether there is a gradient of score with respect to each argument
# it returns 'nothing' for those arguemnts that don't have a derivatice
has_argument_grads(gen::DynamicDSLFunction) = gen.has_argument_grads

# if it is true, then it expects a value, otherwise it expects 'nothing'
accepts_output_grad(gen::DynamicDSLFunction) = gen.accepts_output_grad

function parse_arg_types(args)
    types = Vector{Any}()
    for arg in args
        if isa(arg, Expr) && arg.head == :(::)
            typ = arg.args[2]
            push!(types, typ)
        elseif isa(arg, Symbol)
            push!(types, :Any)
        else
            error("Error parsing argument to generative function: $arg")
        end
    end
    types
end

macro ad(ast)
    if (!isa(ast, Expr)
        || ast.head != :macrocall
        || ast.args[1] != Symbol("@gen")
        || length(ast.args) != 3
        || !isa(ast.args[2], LineNumberNode))
        error("Syntax error in @ad $ast")
    end
    parse_gen_function(ast.args[3], true)
end

macro addr(expr::Expr, addr, addrdiff)
    if expr.head != :call
        error("syntax error in @addr at $(expr)")
    end
    fn = esc(expr.args[1])
    args = map(esc, expr.args[2:end])
    Expr(:call, :addr, esc(state), fn, Expr(:tuple, args...), esc(addr), esc(addrdiff))
end

macro gen(ast)
    parse_gen_function(ast, false)
end

function args_for_gf_julia_fn(args, has_argument_grads)
    # if an argument is marked for AD, we change remove the type annotation for
    # the Julia function implementation, because we use reverse mode AD using
    # boxed tracked values. we don't give any type information about the argument.
    escaped_args = []
    for (arg, has_grad) in zip(args, has_argument_grads)
        if has_grad
            if isa(arg, Expr) && arg.head == :(::)
                push!(escaped_args, esc(arg.args[1]))
            elseif isa(arg, Symbol)
                push!(escaped_args, esc(arg))
            end
        else
            push!(escaped_args, esc(arg))
        end
    end
    escaped_args
end

transform_body_for_non_update(ast) = ast

function transform_body_for_non_update(ast::Expr)
    @assert !(ast.head == :macrocall && ast.args[1] == Symbol("@diff"))

    # remove the argdiff argument from @addr expressions
    if ast.head == :macrocall && ast.args[1] == Symbol("@addr")
        if length(ast.args) == 5
            @assert isa(ast.args[2], LineNumberNode)
            # remove the last argument
            ast = Expr(ast.head, ast.args[1:4]...)
        end
    end

    # remove any @diff sub-expressions, and recurse
    new_args = Vector()
    for arg in ast.args
        if !(isa(arg, Expr) && arg.head == :macrocall && arg.args[1] == Symbol("@diff"))
            push!(new_args, transform_body_for_non_update(arg))
        end
    end
    Expr(ast.head, new_args...)
end

transform_body_for_update(ast, in_diff) = ast

function transform_body_for_update(ast::Expr, in_diff::Bool)
    @assert !(ast.head == :macrocall && ast.args[1] == Symbol("@diff"))
    new_args = Vector()
    for arg in ast.args
        if isa(arg, Expr) && arg.head == :macrocall && arg.args[1] == Symbol("@diff")
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


function parse_gen_function(ast, ad_annotation::Bool)
    if ast.head != :function
        error("syntax error at $(ast) in $(ast.head)")
    end
    if length(ast.args) != 2
        error("syntax error at $(ast) in $(ast.args)")
    end
    signature = ast.args[1]
    body = ast.args[2]
    if signature.head != :call
        error("syntax error at $(ast) in $(signature)")
    end
    args = signature.args[2:end]
    has_argument_grads = map(marked_for_ad, args)
    args = map(strip_marked_for_ad, args)

    # julia function definition (for a non-update behavior)
    escaped_args = args_for_gf_julia_fn(args, has_argument_grads)
    gf_args = [esc(state), escaped_args...]
    julia_fn_defn = Expr(:function,
        Expr(:tuple, gf_args...),
        esc(transform_body_for_non_update(body)))

    # julia function definition for update function
    update_julia_fn_defn = Expr(:function,
        Expr(:tuple, gf_args...),
        esc(transform_body_for_update(body, false)))

    # create generator and assign it a name
    generator_name = signature.args[1]
    arg_types = map(esc, parse_arg_types(args))
    Expr(:block,
        Expr(:(=), 
            esc(generator_name),
            Expr(:call, :DynamicDSLFunction,
                quote Type[$(arg_types...)] end,
                julia_fn_defn, update_julia_fn_defn,
                has_argument_grads, ad_annotation)))
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

#########
# delta #
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

##################
# AddressVisitor #
##################

struct AddressVisitor
    visited::DynamicAddressSet
end

AddressVisitor() = AddressVisitor(DynamicAddressSet())

function visit!(visitor::AddressVisitor, addr)
    push_leaf_node!(visitor.visited, addr)
end

function _diff(trie, addrs::AddressSet)
    diff_trie = HomogenousTrie{Any,Any}()

    for (key_field, value) in get_leaf_nodes(trie)
        if !has_leaf_node(addrs, key_field)
            set_leaf_node!(diff_trie, key_field, value)
        end
    end

    for (key_field, node) in get_internal_nodes(trie)
        if !has_leaf_node(addrs, key_field)
            if has_internal_node(addrs, key_field)
                sub_addrs = get_internal_node(addrs, key_field)
            else
                sub_addrs = EmptyAddressSet()
            end
            diff_node = _diff(node, sub_addrs)
            if !isempty(diff_node)
                set_internal_node!(diff_trie, key_field, diff_node)
            end
        end
    end

    diff_trie
end

function get_unvisited(visitor::AddressVisitor, choices)
    _diff(choices, visitor.visited)
end

function lightweight_check_no_internal_node(constraints::Assignment, addr)
    if has_internal_node(constraints, addr)
        error("Expected a value at address $addr but trace contained an assignment")
    end
end

function lightweight_check_no_leaf_node(constraints::Assignment, addr)
    if has_leaf_node(constraints, addr)
        error("Expected an assignment at address $addr but trace contained a value")
    end
end

function lightweight_retchange_already_set_err()
    error("@retdiff! was already used")
end

include("generate.jl")
include("update.jl")
include("project.jl")
include("ungenerate.jl")
include("backprop_params.jl")
include("backprop_trace.jl")

export set_param!, get_param, get_param_grad, zero_param_grad!, init_param!
export @ad
export @delta
export @param
export @addr
export @gen
export @choicediff, @calldiff
export @argdiff
export @retdiff
