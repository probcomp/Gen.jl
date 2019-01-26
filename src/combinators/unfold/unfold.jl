using FunctionalCollections: PersistentVector, push, assoc

struct UnfoldType end

"""
    gen_fn = Unfold(kernel::GenerativeFunction)

Return a new generative function that applies the kernel in sequence, passing the return value of one application as an input to the next.

The kernel accepts the following arguments:

- The first argument is the `Int` index indicating the position in the sequence (starting from 1).

- The second argument is the *state*.

- The kernel may have additional arguments after the state.

The return type of the kernel must be the same type as the state.

The returned generative function accepts the following arguments:

- The number of times (N) to apply the kernel.

- The initial state.

- The rest of the arguments (not including the state) that will be passed to each kernel application.

The return type of the returned generative function is `FunctionalCollections.PersistentVector{T}` where `T` is the return type of the kernel.
"""
struct Unfold{T,U} <: GenerativeFunction{PersistentVector{T},VectorTrace{UnfoldType,T,U}}
    kernel::GenerativeFunction{T,U}
end

export Unfold

function has_argument_grads(gen_fn::Unfold)
    (false, has_argument_grads(gen_fn.kernel)[2:end]...)
end

# TODO
accepts_output_grad(gen_fn::Unfold) = false

"""
    argdiff = UnfoldCustomArgDiff(init_changed::Bool, params_changed::Bool)

Construct an argdiff that indicates whether the initial state may have changed (`init_changed`) , and whether or not the remaining arguments to the kernel may have changed (`params_changed`).
"""
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
