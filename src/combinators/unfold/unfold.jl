using FunctionalCollections: PersistentVector, push, assoc

struct UnfoldType end

# accepts as argdiff: UnfoldCustomArgDiff, NoArgDiff, UnknownArgDiff
# returns for retdiff: NoRetDiff, VectorCustomRetDiff
struct Unfold{T,U} <: GenerativeFunction{PersistentVector{T},VectorTrace{UnfoldType,T,U}}
    kernel::GenerativeFunction{T,U}
end

export Unfold

function has_argument_grads(gen_fn::Unfold)
    (false, has_argument_grads(gen_fn.kernel)[2:end]...)
end

# TODO
accepts_output_grad(gen_fn::Unfold) = false

struct UnfoldCustomArgDiff
    init_changed::Bool
    params_changed::Bool
end

export UnfoldCustomArgDiff

function unpack_args(args::Tuple)
    len = args[1]
    init_state = args[2]
    params = args[3:end]
    (len, init_state, params)
end

function check_length(len::Int)
    if len < 0
        error("unfold got length of $len < 0")
    end
end

include("initialize.jl")
include("propose.jl")
include("assess.jl")
include("generic_update.jl")
include("force_update.jl")
include("fix_update.jl")
include("free_update.jl")
include("extend.jl")
include("backprop.jl")
