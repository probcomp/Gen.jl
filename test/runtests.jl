using Gen
using Test
import Random

"""
Compute a numerical partial derivative of `f` with respect to the `i`th
argument using finite differences.

If `broadcast` is `false`, then `args[i]` must be a scalar.

If `broadcast` is `true`, then `args[i]` may be an array.  In that case, `f` is
assumed to be the broadcast of a function whose `i`th argument is a scalar.  In
particular, `f` must still operate independently on each element of `args[i]`.
This condition cannot be checked automatically for arbitrary `f::Function`, so
the caller must guarantee it.
"""
function finite_diff(f::Function, args::Tuple, i::Int, dx::Float64;
                     broadcast=false)
    pos_args = Any[args...]
    neg_args = Any[args...]
    if broadcast
        pos_args[i] = copy(args[i]) .+ dx
        neg_args[i] = copy(args[i]) .- dx
        return (f(pos_args...) - f(neg_args...)) ./ (2. * dx)
    else
        pos_args[i] += dx
        neg_args[i] -= dx
        return (f(pos_args...) - f(neg_args...)) / (2. * dx)
    end
end

function finite_diff_vec(f::Function, args::Tuple, i::Int, j::Int, dx::Float64)
    pos_args = Any[deepcopy(args)...]
    pos_args[i][j] += dx
    neg_args = Any[deepcopy(args)...]
    neg_args[i][j] -= dx
    return (f(pos_args...) - f(neg_args...)) / (2. * dx)
end

"""
Compute a numerical partial derivative of `f` with respect to the `i`th argument, a symettric matrix, using finite differences.

Finite differences are applied symettrically to off-diagonal elements of the matrix, ensuring that the modified matrix is still symettric.

When applied to off-diagonal elements, the finite difference is scaled by a factor of 1/2.

The condition that the `i`th argument is a symettric matrix cannot be checked automatically, so the caller must guarantee it.
"""

function finite_diff_mat_sym(f::Function, args::Tuple, i::Int, j::Int, k::Int, dx::Float64)
    pos_args = Any[deepcopy(args)...]
    pos_args[i][j, k] += dx
    neg_args = Any[deepcopy(args)...]
    neg_args[i][j, k] -= dx

    if j!=k
        pos_args[i][k, j] += dx
        neg_args[i][k, j] -= dx
        return (f(pos_args...) - f(neg_args...)) / (4. * dx)
    end

    return (f(pos_args...) - f(neg_args...)) / (2. * dx)
end

"Macro to test nesting of macros in @trace calls."
macro insert_normal_call(x)
    :(normal($(esc(x)), 0.0))
end

const dx = 1e-6

include("autodiff.jl")
include("diff.jl")
include("selection.jl")
include("assignment.jl")
include("dynamic_dsl.jl")
include("optional_args.jl")
include("static_ir/static_ir.jl")
include("static_dsl.jl")
include("tilde_sugar.jl")
include("inference/inference.jl")
include("modeling_library/modeling_library.jl")