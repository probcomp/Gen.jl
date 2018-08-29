###################
# plate generator # 
###################

"""
Generator that makes many independent application of a kernel generator,
similar to 'map'.  The arguments are a tuple of vectors, each of of length N,
where N is the nubmer of applications of the kernel.
"""
struct Plate{T,U} <: Generator{PersistentVector{T},VectorTrace{T,U}}
    kernel::Generator{T,U}
end

accepts_output_grad(plate::Plate) = accepts_output_grad(plate.kernel)
has_argument_grads(plate::Plate) = has_argument_grads(plate.kernel)

function plate(kernel::Generator{T,U}) where {T,U}
    Plate{T,U}(kernel)
end

function get_static_argument_types(plate::Plate)
    [Expr(:curly, :Vector, typ) for typ in get_static_argument_types(plate.kernel)]
end

function get_args_for_key(args::Tuple, key::Int)
    map((arg) -> arg[key], args)
end

function get_prev_and_new_lengths(args, trace)
    new_length = length(args[1])
    prev_args = get_call_record(trace).args
    prev_length = length(prev_args[1])
    (new_length, prev_length)
end

struct PlateChange{T}

    # the subset of kernel args which may have changed
    changed_args::Vector{Int}
 
    # the deltas to pass to each changed invocation of the kernel
    sub_changes::Vector{T}
end

include("simulate.jl")
include("assess.jl")
include("generate.jl")
include("update.jl")
include("extend.jl")
include("project.jl")
#include("backprop_params.jl")
include("backprop_trace.jl")

#include("ungenerate.jl")

export plate
export PlateChange
