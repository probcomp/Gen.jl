import Zygote

#######################
# normal distribution #
#######################

abstract type Distribution end

struct Normal <: Distribution end

const normal = Normal()

function simulate(::Normal, mu, std)
    (randn() * std) + mu
end

function logpdf(::Normal, x, mu, std)
     var = std * std
    diff = x - mu
    -(diff * diff)/ (2.0 * var) - 0.5 * log(2.0 * pi * var)
end

#######################
# code transformation #
#######################

struct GenFunction
    code::Function # the first argument of this function is the trace
end

const state = gensym("state")

struct Argument
    name::Symbol
    typ::Union{Symbol,Expr}
end

function parse_arg(expr)
    if isa(expr, Symbol)
        # x
        arg = Argument(expr, :Any)
    elseif isa(expr, Expr) && expr.head == :(::)
        # x::Int
        arg = Argument(expr.args[1], expr.args[2])
    else
        dump(expr)
        error("syntax error in gen function argument at $expr")
    end
    arg
end

macro gen(ast)
    signature = ast.args[1]
    body = ast.args[2]
    @assert signature.head == :call
    name = signature.args[1]
    args = map(parse_arg, signature.args[2:end])
    escaped_args = map((arg) -> esc(arg.name), args)
    gf_args = [esc(state), escaped_args...]
    julia_fn_name = gensym(name)
    julia_fn_defn = Expr(:function,
        Expr(:call, esc(julia_fn_name), gf_args...),
        esc(body))
    quote
        $(esc(name)) = GenFunction($julia_fn_defn)
    end
end

macro record(dist, args, addr)
    @assert args.head == :tuple
    args = map(esc, args.args)
    quote
        record!($(esc(state)), $(esc(dist)), $(Expr(:tuple, args...)), $(esc(addr)))
    end
end

############
# simulate #
############

struct SimulateState
    trace::Dict
end

function record!(state::SimulateState, dist::Distribution, args, addr)   
    if haskey(state.trace, addr)
        error("Address already visited: $addr")
    end
    value = simulate(dist, args...)
    state.trace[addr] = value
    value
end

function simulate(gf::GenFunction, args::Tuple)
    state = SimulateState(Dict())
    retval = gf.code(state, args...)
    (state.trace, retval)
end

############
# backprop #
############

mutable struct BackpropState
    trace::Dict
    logpdf::Float64
end

function record!(state::BackpropState, dist::Distribution, args, addr)   
    @assert haskey(state.trace, addr)
    value = state.trace[addr]
    state.logpdf += logpdf(dist, value, args...)
    value
end

function trace_logpdf(gf::GenFunction, trace::Dict, args)
    state = BackpropState(trace, 0.)
    gf.code(state, args...)#args[1], args[2])
    state.logpdf
end

function backprop(gf::GenFunction, trace::Dict, args, retgrad::Nothing)
    lpdf, back = Zygote.pullback(trace_logpdf, gf, trace, args)
    back(1.)
end

function backprop(gf::GenFunction, trace::Dict, args, retgrad)
    lpdf, back = Zygote.pullback(trace_logpdf, gf, trace, args)
    back(1.)
end


# TODO handle return value grad for compositional AD

# TODO implement custom adjoint to support calling a (black box) generative function

###########
# example #
###########


println(macroexpand(Main, :(@gen function foo(a, y)
    z = @record(normal, (a, 1), :z)
    return y + z
end)))

@gen function foo(a, y)
    z = @record(normal, (a, 1), :z)
    return y + z
end

trace, r = simulate(foo, (1, 2))
println(r)
println(trace[:z])

grad = backprop(foo, trace, (1.0, 2.0))
println(grad)
