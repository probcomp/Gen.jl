# TODO: This does not allow a distribution to change type.
# (i.e., from Int to Float64).
using MacroTools

# A function call within a @dist body
# TODO: Improve to handle when args includes an Arg.
# Will require creating a new type of Arg.
dist_call(f, args) = f(args...)
dist_call(d::Distribution{T}, args) where T = DistWithArgs{T}(d, args)


#TODO: Make this actually compile a new type
macro dist(fnexpr)
  # First, pull out arguments and body
  fndef = splitdef(longdef(fnexpr))
  #
  # if !@capture(fnexpr, argexpr_ -> body_)
  #     error("@dist expression must be of the form @dist (args...) -> body")
  # end
  # arguments = (typeof(argexpr) == Symbol) ? (argexpr,) : argexpr.args # [] #argexpr

  arguments = fndef[:args]
  body = fndef[:body]

  name_to_index = Dict{Symbol, Int8}()
  for (i, arg) in enumerate(arguments)
      (argname, _, _, _) = splitarg(arg)
      name_to_index[argname] = i
  end

  function process_node(node)
      if typeof(node) == Symbol &&haskey(name_to_index, node)
          return :($(Arg(name_to_index[node])))
      end
      if @capture(node, f_(xs__)) && f != :dist_call
          return :(Gen.dist_call($(f), ($(xs...),)))
      end
      node
  end

  dwa_expr = MacroTools.postwalk(process_node, body)
  :($(esc(fndef[:name])) = compile_dist_with_args($(esc(dwa_expr)), Int8($(length(arguments)))))
end

# Represents a argument at a certain index
struct Arg
    i :: Int8
end

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
    mapping = Dict{Int8, Int8}()
    for (i, arg) in enumerate(d.arglist)
        if typeof(arg) == Arg
            mapping[arg.i] = Int8(i)
        end
    end
    base_arg_grads = has_argument_grads(d.base)
    arg_grad_bools = fill(true, n)
    for (i, arg) in enumerate(d.arglist)
        if typeof(arg) == Arg
            arg_grad_bools[arg.i] = arg_grad_bools[arg.i] && base_arg_grads[i]
        end
    end
    CompiledDistWithArgs{T}(d.base, n, arg_grad_bools, d.arglist)
end

eval_arg(x::Any, args) = x
eval_arg(x::Arg, args) = args[x.i]

function logpdf(d::CompiledDistWithArgs{T}, x::T, args...) where T
    concrete_args = [eval_arg(arg, args) for arg in d.arglist]
    logpdf(d.base, x, concrete_args...)
end

function logpdf_grad(d::CompiledDistWithArgs{T}, x::T, args...) where T
    concrete_args = [eval_arg(arg, args) for arg in d.arglist]
    base_grad = logpdf_grad(d.base, x, concrete_args...)

    # Look into whether we can change this...
    this_grad::Vector{Any} = fill(missing, d.n_args+1)
    # output grad
    this_grad[1] = base_grad[1]
    # arg grads
    for (argind, arg) in enumerate(d.arglist)
        if typeof(arg) == Arg
            if this_grad[arg.i+1] === nothing || base_grad[argind+1] === nothing
                this_grad[arg.i+1] = nothing
            elseif ismissing(this_grad[arg.i+1])
                this_grad[arg.i+1] = base_grad[argind+1]
            else
                this_grad[arg.i+1] += base_grad[argind+1]
            end
        end
    end
    this_grad
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

struct TransformedDistribution{T, U} <: Distribution{T}
    base :: Distribution{U}
    # How many more parameters does this distribution have
    # than the base distribution?
    nArgs :: Int8
    # forward is a U, arg... -> T function,
    # and backward is a T, arg... -> U function,
    # such that for any `args`, we have
    # backward(forward(u, args...), args...) == u
    # and
    # forward(backward(t, args...), args...) == t.
    # Note that if base is a continuous distribution, then
    # forward and backward must be differentiable.
    forward :: Function
    backward :: Function
    backward_grad :: Function
end

function random(d::TransformedDistribution{T, U}, args...)::T where {T, U}
    d.forward(random(d.base, args[d.nArgs+1:end]...), args[1:d.nArgs]...)
end

function logpdf_correction(d::TransformedDistribution{T, U}, x, args) where {T, U}
    log(abs(d.backward_grad(x, args...)[1]))
end

function logpdf(d::TransformedDistribution{T, U}, x::T, args...) where {T, U}
    orig_x = d.backward(x, args[1:d.nArgs]...)
    orig_logpdf = logpdf(d.base, orig_x, args[d.nArgs+1:end]...)

    if is_discrete(d.base)
        orig_logpdf
    else
        orig_logpdf + logpdf_correction(d, x, args[1:d.nArgs])
    end
end

function logpdf_grad(d::TransformedDistribution{T, U}, x::T, args...) where {T, U}
    orig_x = d.backward(x, args[1:d.nArgs]...)
    base_grad = logpdf_grad(d.base, orig_x, args[d.nArgs+1:end]...)

    if is_discrete(d.base) || !has_output_grad(d.base)
        # TODO: should this be nothing or 0?
        [base_grad[1], fill(nothing, d.nArgs)..., base_grad[2:end]...]
    else
        transformation_grad = d.backward_grad(x, args[1:d.nArgs]...)
        correction_grad = ReverseDiff.gradient(v -> logpdf_correction(d, v[1], v[2:end]), [x, args[1:d.nArgs]...])
        # TODO: Will this sort of thing work if the arguments w.r.t. which we are taking
        # gradients are themselves vector-valued?
        full_grad = (transformation_grad .* base_grad[1]) .+ correction_grad
        [full_grad..., base_grad[2:end]...]
    end
end

is_discrete(d::TransformedDistribution{T, U}) where {T, U} = is_discrete(d.base)

(d::TransformedDistribution{T, U})(args...) where {T, U} = random(d, args...)

function has_output_grad(d::TransformedDistribution{T, U}) where {T, U}
    has_output_grad(d.base)
end

function has_argument_grads(d::TransformedDistribution{T, U}) where {T, U}
    if is_discrete(d.base) || !has_output_grad(d.base)
        [fill(false, d.nArgs)..., has_argument_grads(d.base)...]
    else
        [fill(true, d.nArgs)..., has_argument_grads(d.base)...]
    end
end

# Addition
Base.:+(b::DistWithArgs{T}, a::Real) where T <: Real = DistWithArgs(TransformedDistribution{T, T}(b.base, 0, x -> x + a, x -> x - a, x -> (1.0,)), b.arglist)
Base.:+(b::DistWithArgs{T}, a::Arg) where T <: Real = DistWithArgs(TransformedDistribution{T, T}(b.base, 1, +, -, (x, a) -> (1.0, -1.0)), (a, b.arglist...))
Base.:+(a::Real, b::DistWithArgs{T}) where T <: Real = b + a
Base.:+(a::Arg, b::DistWithArgs{T}) where T <: Real = b + a

# Subtraction
Base.:-(b::DistWithArgs{T}, a::Real) where T <: Real = b + (-1 * a)
Base.:-(b::DistWithArgs{T}, a::Arg) where T <: Real = DistWithArgs(TransformedDistribution{T, T}(b.base, 1, -, +, (x, a) -> (1.0, 1.0)), (a, b.arglist...))
Base.:-(a::Real, b::DistWithArgs{T}) where T <: Real = DistWithArgs(TransformedDistribution{T, T}(b.base, 1, x -> a - x, x -> a - x, x -> (-1.0,)), b.arglist)
Base.:-(a::Arg, b::DistWithArgs{T}) where T <: Real = DistWithArgs(TransformedDistribution{T, T}(b.base, 1, (x, a) -> a - x, (x, a) -> a - x, (x, a) -> (-1.0, 1.0)), (a, b.arglist...))

# Multiplication
Base.:*(b::DistWithArgs{T}, a::Real) where T <: Real = DistWithArgs(TransformedDistribution{T, T}(b.base, 0, x -> x * a, x -> x / a, x -> (1.0/a,)), b.arglist)
Base.:*(b::DistWithArgs{T}, a::Arg) where T <: Real = DistWithArgs(TransformedDistribution{T, T}(b.base, 1, *, /, (x, a) -> (1.0/a, -x/(a*a))), (a, b.arglist...))
Base.:*(a::Real, b::DistWithArgs{T}) where T <: Real = b * a
Base.:*(a::Arg, b::DistWithArgs{T}) where T <: Real = b * a

# Division
Base.:/(b::DistWithArgs{T}, a::Real) where T <: Real = DistWithArgs(TransformedDistribution{T, T}(b.base, 0, x -> x / a, x -> x * a, x -> (a,)), b.arglist)
Base.:/(b::DistWithArgs{T}, a::Arg) where T <: Real = DistWithArgs(TransformedDistribution{T, T}(b.base, 1, /, *, (x, a) -> (a, x)), (a, b.arglist...))
Base.:/(a::Real, b::DistWithArgs{T}) where T <: Real = DistWithArgs(TransformedDistribution{T, T}(b.base, 0, x -> a / x, x -> a / x, x -> (-a / (x*x),)), b.arglist)
Base.:/(a::Arg, b::DistWithArgs{T}) where T <: Real = DistWithArgs(TransformedDistribution{T, T}(b.base, 1, (x, a) -> a/x, (x, a) -> a/x, (x, a) -> (-a/(x*x), 1.0/x)), (a, b.arglist...))

# Exponentiation
Base.exp(b::DistWithArgs{T}) where T <: Real = DistWithArgs(TransformedDistribution{T, T}(b.base, 0, exp, log, x -> (1.0 / x,)), b.arglist)
Base.log(b::DistWithArgs{T}) where T <: Real = DistWithArgs(TransformedDistribution{T, T}(b.base, 0, log, exp, x -> (exp(x),)), b.arglist)

struct WithLabelArg{T, U} <: Distribution{T}
    base :: Distribution{U}
end

function logpdf(d::WithLabelArg{T, U}, x::T, collection, base_args...) where {T, U}
    if is_discrete(d.base)
        # Accumulate
        logprobs::Array{Float64, 1} = []
        for p in pairs(collection)
            (index, item) = (p.first, p.second)
            if item == x
                push!(logprobs, logpdf(d.base, index, base_args...))
            end
        end
        logsumexp(logprobs)
    else
        error("Cannot relabel a continuous distribution")
    end
end

function logpdf_grad(d::WithLabelArg{T, U}, x::T, collection, base_args...) where {T, U}
    base_arg_grads = fill(nothing, length(base_args))

    for p in pairs(collection)
        (index, item) = (p.first, p.second)
        if item == x
            new_grads = logpdf_grad(d.base, index, base_args...)
            for (arg_idx, grad) in enumerate(new_grads)
                if base_arg_grads[arg_idx] === nothing
                    base_arg_grads[arg_idx] = grad
                elseif grad !== nothing
                    base_arg_grads[arg_idx] += grad
                end
            end
        end
    end
    (nothing, nothing, base_arg_grads...)
end

function random(d::WithLabelArg{T, U}, collection, base_args...)::T where {T, U}
    collection[random(d.base, base_args...)]
end

is_discrete(d::WithLabelArg{T, U}) where {T, U} = true

(d::WithLabelArg{T, U})(collection, base_args...) where {T, U} = random(d, collection, base_args...)

function has_output_grad(d::WithLabelArg{T, U}) where {T, U}
    false
end

has_argument_grads(d::WithLabelArg{T, U}) where {T, U} = (false, has_argument_grads(d.base)...)

Base.getindex(collection::Arg, d::DistWithArgs{T}) where T = DistWithArgs{Any}(WithLabelArg{Any, T}(d.base), (collection, d.arglist...))

struct RelabeledDistribution{T, U} <: Distribution{T}
    base :: Distribution{U}
    collection::Union{AbstractArray{T}, AbstractDict{T}}
end

function logpdf(d::RelabeledDistribution{T, U}, x::T, base_args...) where {T, U}
    if is_discrete(d.base)
        # Accumulate
        logprobs::Array{Float64, 1} = []
        for p in pairs(d.collection)
            (index, item) = (p.first, p.second)
            if item == x
                push!(logprobs, logpdf(d.base, index, base_args...))
            end
        end
        logsumexp(logprobs)
    else
        error("Cannot relabel a continuous distribution")
    end
end

function logpdf_grad(d::RelabeledDistribution{T, U}, x::T, base_args...) where {T, U}
    base_arg_grads = fill(nothing, length(base_args))

    for p in pairs(d.collection)
        (index, item) = (p.first, p.second)
        if item == x
            new_grads = logpdf_grad(d.base, index, base_args...)
            for (arg_idx, grad) in enumerate(new_grads)
                if base_arg_grads[arg_idx] === nothing
                    base_arg_grads[arg_idx] = grad
                elseif grad !== nothing
                    base_arg_grads[arg_idx] += grad
                end
            end
        end
    end
    (nothing, base_arg_grads...)
end

function random(d::RelabeledDistribution{T, U}, base_args...)::T where {T, U}
    d.collection[random(d.base, base_args...)]
end

is_discrete(d::RelabeledDistribution{T, U}) where {T, U} = true

(d::RelabeledDistribution{T, U})(base_args...) where {T, U} = random(d, base_args...)

function has_output_grad(d::RelabeledDistribution{T, U}) where {T, U}
    false
end

has_argument_grads(d::RelabeledDistribution{T, U}) where {T, U} = has_argument_grads(d.base)

Base.getindex(collection::AbstractArray{T}, d::DistWithArgs{U}) where {T, U} = DistWithArgs{T}(RelabeledDistribution{T, U}(d.base, collection), d.arglist)
Base.getindex(collection::AbstractDict{T}, d::DistWithArgs{U}) where {T, U} = DistWithArgs{T}(RelabeledDistribution{T, U}(d.base, collection), d.arglist)

export @dist



# TODO: Gradient calculation is wrong if the argument appears more than once.
#  We should _sum_ contributions.

# The problem with f(a) is for derivative taking.
