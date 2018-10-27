###################
# plate generator # 
###################

"""
Generator that makes many independent application of a kernel generator,
similar to 'map'.  The arguments are a tuple of vectors, each of of length N,
where N is the nubmer of applications of the kernel.
"""
struct Plate{T,U,V,W} <: Generator{PersistentVector{T},VectorTrace{T,U}}
    kernel::Generator{T,U}
    kernel_noargdiff::V
    kernel_unknownargdiff::W
end

accepts_output_grad(plate::Plate) = accepts_output_grad(plate.kernel)
has_argument_grads(plate::Plate) = has_argument_grads(plate.kernel)

function plate(kernel::Generator{T,U}, noargdiff::V=, unknownargdiff::W=) where {T,U,V,W}
    Plate{T,U,V,W}(kernel, noargdiff, unknownargdiff)
end

function get_static_argument_types(plate::Plate)
    [Vector{typ} for typ in get_static_argument_types(plate.kernel)]
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

export plate

###########
# argdiff #
###########

# accepts: NoArgDiff, UnknownArgDiff, and PlateCustomArgDiff

"""
The number of applications may have changed. Custom argdiffs are provided for
retained applications whose arguments may have changed.
"""
struct PlateCustomArgDiff{T}
    retained_argdiffs::Dict{Int,T} 
end

# if there is no argdiff provided, 

# NOTE: the kernel function must accept noargdiff and unknownargdiff as argdiffs


###########
# retdiff #
###########

"""
The return value of the plate has not changed.
"""
struct PlateNoRetDiff end
isnoretdiff(::PlateNoRetDiff) = true

"""
The number of applications may have changed. retdiff values are provided for
retained applications for which isnoretdiff() = false.
"""
struct PlateCustomRetDiff{T}
    retained_retdiffs::Dict{Int,T}
end
isnoretdiff(::PlateCustomRetDiff) = false



###############################
# generator interface methods #
###############################

include("simulate.jl")
include("assess.jl")
include("generate.jl")
include("update.jl")
include("extend.jl")
include("project.jl")
include("backprop_trace.jl")

