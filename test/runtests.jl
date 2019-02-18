using Gen
using Test
import Random

function finite_diff(f::Function, args::Tuple, i::Int, dx::Float64)
    pos_args = Any[args...]
    pos_args[i] += dx
    neg_args = Any[args...]
    neg_args[i] -= dx
    return (f(pos_args...) - f(neg_args...)) / (2. * dx)
end

function finite_diff_vec(f::Function, args::Tuple, i::Int, j::Int, dx::Float64)
    pos_args = Any[deepcopy(args)...]
    pos_args[i][j] += dx
    neg_args = Any[deepcopy(args)...]
    neg_args[i][j] -= dx
    return (f(pos_args...) - f(neg_args...)) / (2. * dx)
end

const dx = 1e-6

include("autodiff.jl")
include("diff.jl")
include("assignment.jl")
include("dynamic_dsl.jl")
include("static_ir.jl")
include("static_dsl.jl")
include("injective.jl")
include("inference/inference.jl")
include("modeling_library/modeling_library.jl")
