using MacroTools

include("transformed_distribution.jl")
include("relabeled_distribution.jl")

#TODO: Remove Distribution{T} from structs; use a type U instead

# `Arg` types: represent arguments to distributions
abstract type Arg{T} end

# Represents a argument at a certain index
struct SimpleArg{T} <: Arg{T}
    i :: Int8
end

struct TransformedArg{T} <: Arg{T}
    f_args :: Vector{Arg} # Is it better to do Union{TransformedArg, SimpleArg}?
    orig_f :: Function
    arg_passer :: Function
end

# Each j is an Arg, and has a yes/no answer about whether
# it supports gradients. The Arg may correspond to multiple i's,
# which are user-facing parameters.
all_indices(arg::SimpleArg) = [arg.i]
all_indices(arg::TransformedArg) = vcat([all_indices(a) for a in arg.f_args]...)

# Evaluate user-facing args to concrete values passed to the base distribution
eval_arg(x::Any, args) = x
eval_arg(x::SimpleArg, args) = typecheck_arg(x, args[x.i])
eval_arg(x::TransformedArg, args) =
    x.arg_passer(x.orig_f, [eval_arg(a, args) for a in x.f_args]...)

# Type of SimpleArg must match arg, otherwise a MethodError will be thrown
typecheck_arg(x::SimpleArg{T}, arg::T) where {T} = arg

# DistWithArgs
struct DistWithArgs{T}
    base :: Distribution{T}
    arglist # Contains Args and other values
end

struct CompiledDistWithArgs{T} <: Distribution{T}
    base :: Distribution{T}
    n_args :: Int8
    arg_grad_bools
    arglist
end

function compile_dist_with_args(d::DistWithArgs{T}, n::Int8)::CompiledDistWithArgs{T} where T
    base_arg_grads = has_argument_grads(d.base)
    arg_grad_bools = fill(true, n)
    for (i, arg) in enumerate(d.arglist)
        if typeof(arg) <: Arg
            for idx in all_indices(arg)
                arg_grad_bools[idx] = arg_grad_bools[idx] && base_arg_grads[i]
            end
        end
    end
    CompiledDistWithArgs{T}(d.base, n, arg_grad_bools, d.arglist)
end

function logpdf(d::CompiledDistWithArgs{T}, x::T, args...) where T
    concrete_args = [eval_arg(arg, args) for arg in d.arglist]
    logpdf(d.base, x, concrete_args...)
end

function logpdf_grad(d::CompiledDistWithArgs{T}, x::T, args...) where T
    self_has_output_grad = has_output_grad(d)
    self_has_arg_grads = has_argument_grads(d)
    concrete_args = [eval_arg(arg, args) for arg in d.arglist]
    base_has_arg_grads = has_argument_grads(d.base)
    base_grads = logpdf_grad(d.base, x, concrete_args...)

    base_arg_grads = [g for (i, g) in enumerate(base_grads[2:end])
                      if base_has_arg_grads[i]]
    argvec = collect(args)
    eval_arg_grads = hcat([ReverseDiff.gradient(xs -> eval_arg(arg, xs), argvec)
        for (i, arg) in enumerate(d.arglist) if base_has_arg_grads[i]]...)

    retval = [base_grads[1]]
    for i in 1:d.n_args
        if self_has_arg_grads[i]
            push!(retval, eval_arg_grads[i,:]' * base_arg_grads)
        else
            push!(retval, nothing)
        end
    end
    retval
end

function random(d::CompiledDistWithArgs{T}, args...)::T where T
    concrete_args = [eval_arg(arg, args) for arg in d.arglist]
    random(d.base, concrete_args...)
end

is_discrete(d::CompiledDistWithArgs{T}) where T = is_discrete(d.base)

(d::CompiledDistWithArgs{T})(args...) where T = random(d, args...)

function has_output_grad(d::CompiledDistWithArgs{T}) where T
    has_output_grad(d.base)
end

function has_argument_grads(d::CompiledDistWithArgs{T}) where T
    d.arg_grad_bools
end

# ACTUAL DSL

# A function call within a @dist body
function dist_call(d::Distribution{T}, args, __module__) where T
    DistWithArgs{T}(d, args)
end

function dist_call(f, args, __module__)
    if any(x -> (typeof(x) <: DistWithArgs), args)
        f(args...)
    elseif any(x -> (typeof(x) <: Arg), args)
        # Create a transformed argument
        f_args = []
        new_f_arg_names = []
        actual_arg_exprs = []
        for (i, x) in enumerate(args)
            if typeof(x) <: Arg
                push!(f_args, x)
                name = gensym()
                push!(new_f_arg_names, name)
                push!(actual_arg_exprs, name)
            else
                push!(actual_arg_exprs, Meta.quot(x))
            end
        end
        arg_passer = __module__.eval(
            :((f, $(new_f_arg_names...)) -> f($(actual_arg_exprs...))))
        TransformedArg{Any}(f_args, f, arg_passer)
    else
        f(args...)
    end
end


#TODO: Make this actually compile a new type
macro dist(fnexpr)
  # First, pull out arguments and body
  fndef = splitdef(longdef(fnexpr))
  arguments = fndef[:args]
  body = fndef[:body]

  name_to_index = Dict{Symbol, Int8}()
  name_to_type = Dict{Symbol, Type}()
  for (i, arg) in enumerate(arguments)
      (argname, argtype, _, _) = splitarg(arg)
      name_to_index[argname] = i
      name_to_type[argname] = __module__.eval(argtype)
  end

  function process_node(node)
      if typeof(node) == Symbol && haskey(name_to_index, node)
          return :($(SimpleArg{name_to_type[node]}(name_to_index[node])))
      end
      if @capture(node, f_(xs__)) && f != :dist_call
          return :(Gen.dist_call($(f), ($(xs...),), $(__module__)))
      end
      node
  end

  dwa_expr = MacroTools.postwalk(process_node, body)
  :($(esc(fndef[:name])) =
    compile_dist_with_args($(esc(dwa_expr)), Int8($(length(arguments)))))
end

# Addition
Base.:+(b::DistWithArgs{T}, a::Real) where T <: Real =
    DistWithArgs(TransformedDistribution{T, T}(b.base, 0, x -> x + a, x -> x - a, x -> (1.0,)), b.arglist)
Base.:+(b::DistWithArgs{T}, a::Arg) where T <: Real =
    DistWithArgs(TransformedDistribution{T, T}(b.base, 1, +, -, (x, a) -> (1.0, -1.0)), (a, b.arglist...))
Base.:+(a::Real, b::DistWithArgs{T}) where T <: Real = b + a
Base.:+(a::Arg, b::DistWithArgs{T}) where T <: Real = b + a

# Subtraction
Base.:-(b::DistWithArgs{T}, a::Real) where T <: Real = b + (-1 * a)
Base.:-(b::DistWithArgs{T}, a::Arg) where T <: Real =
    DistWithArgs(TransformedDistribution{T, T}(b.base, 1, -, +, (x, a) -> (1.0, 1.0)), (a, b.arglist...))
Base.:-(a::Real, b::DistWithArgs{T}) where T <: Real =
    DistWithArgs(TransformedDistribution{T, T}(b.base, 0, x -> a - x, x -> a - x, x -> (-1.0,)), b.arglist)
Base.:-(a::Arg, b::DistWithArgs{T}) where T <: Real =
    DistWithArgs(TransformedDistribution{T, T}(b.base, 1, (x, a) -> a - x, (x, a) -> a - x, (x, a) -> (-1.0, 1.0)), (a, b.arglist...))

# Multiplication
Base.:*(b::DistWithArgs{T}, a::Real) where T <: Real =
    DistWithArgs(TransformedDistribution{T, T}(b.base, 0, x -> x * a, x -> x / a, x -> (1.0/a,)), b.arglist)
Base.:*(b::DistWithArgs{T}, a::Arg) where T <: Real =
    DistWithArgs(TransformedDistribution{T, T}(b.base, 1, *, /, (x, a) -> (1.0/a, -x/(a*a))), (a, b.arglist...))
Base.:*(a::Real, b::DistWithArgs{T}) where T <: Real = b * a
Base.:*(a::Arg, b::DistWithArgs{T}) where T <: Real = b * a

# Division
Base.:/(b::DistWithArgs{T}, a::Real) where T <: Real =
    DistWithArgs(TransformedDistribution{T, T}(b.base, 0, x -> x / a, x -> x * a, x -> (a,)), b.arglist)
Base.:/(b::DistWithArgs{T}, a::Arg) where T <: Real =
    DistWithArgs(TransformedDistribution{T, T}(b.base, 1, /, *, (x, a) -> (a, x)), (a, b.arglist...))
Base.:/(a::Real, b::DistWithArgs{T}) where T <: Real =
    DistWithArgs(TransformedDistribution{T, T}(b.base, 0, x -> a / x, x -> a / x, x -> (-a / (x*x),)), b.arglist)
Base.:/(a::Arg, b::DistWithArgs{T}) where T <: Real =
    DistWithArgs(TransformedDistribution{T, T}(b.base, 1, (x, a) -> a/x, (x, a) -> a/x, (x, a) -> (-a/(x*x), 1.0/x)), (a, b.arglist...))

# Exponentiation
Base.exp(b::DistWithArgs{T}) where T <: Real =
    DistWithArgs(TransformedDistribution{T, T}(b.base, 0, exp, log, x -> (1.0 / x,)), b.arglist)
Base.log(b::DistWithArgs{T}) where T <: Real =
    DistWithArgs(TransformedDistribution{T, T}(b.base, 0, log, exp, x -> (exp(x),)), b.arglist)

# Indexing
Base.getindex(collection::Arg, d::DistWithArgs{T}) where {T} =
    DistWithArgs{Any}(WithLabelArg{Any, T}(d.base), (collection, d.arglist...))
Base.getindex(collection::Arg{<:AbstractArray{T}}, d::DistWithArgs{U}) where {T, U} =
    DistWithArgs{T}(WithLabelArg{T, U}(d.base), (collection, d.arglist...))
Base.getindex(collection::Arg{<:AbstractDict{U, T}}, d::DistWithArgs{U}) where {T, U} =
    DistWithArgs{T}(WithLabelArg{T, U}(d.base), (collection, d.arglist...))
Base.getindex(collection::AbstractArray{T}, d::DistWithArgs{U}) where {T, U} =
    DistWithArgs{T}(RelabeledDistribution{T, U}(d.base, collection), d.arglist)
Base.getindex(collection::AbstractDict{U, T}, d::DistWithArgs{U}) where {T, U} =
    DistWithArgs{T}(RelabeledDistribution{T, U}(d.base, collection), d.arglist)
# Ensure no ambiguity with `Base` implementation.
Base.getindex(collection::Dict{U, T}, d::DistWithArgs{U}) where {T, U} =
    DistWithArgs{T}(RelabeledDistribution{T, U}(d.base, collection), d.arglist)

# `Enum` constructors
function (::Type{T})(d::DistWithArgs{U}) where {T <: Enum, U}
    lookup = Dict(Int(i) => i for i in instances(T))
    DistWithArgs{T}(RelabeledDistribution{T, U}(d.base, lookup), d.arglist)
end

export @dist
