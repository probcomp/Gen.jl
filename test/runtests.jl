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

const dx = 1e-6

include("autodiff.jl")
include("assignment.jl")
include("lightweight.jl")
include("static_ir.jl")
include("static_dsl.jl")
include("injective.jl")
include("inference.jl")
include("distribution.jl")
include("map.jl")
include("recurse.jl")
