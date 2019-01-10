##################
# map combinator # 
##################

# used for type dispatch on the VectorTrace type (e.g. we will also have a UnfoldType)
struct MapType end 

"""
GenerativeFunction that makes many independent application of a kernel generator,
similar to 'map'.  The arguments are a tuple of vectors, each of of length N,
where N is the nubmer of applications of the kernel.
"""
struct Map{T,U} <: GenerativeFunction{PersistentVector{T},VectorTrace{MapType,T,U}}
    kernel::GenerativeFunction{T,U}
end

export Map

has_argument_grads(map_gf::Map) = has_argument_grads(map_gf.kernel)
accepts_output_grad(map_gf::Map) = accepts_output_grad(map_gf.kernel)

function get_args_for_key(args::Tuple, key::Int)
    map((arg) -> arg[key], args)
end

function get_prev_and_new_lengths(args::Tuple, prev_trace)
    new_length = length(args[1])
    prev_args = get_args(prev_trace)
    prev_length = length(prev_args[1])
    (new_length, prev_length)
end

###########
# argdiff #
###########

# accepts: NoArgDiff, UnknownArgDiff, and MapCustomArgDiff

"""
The number of applications may have changed. Custom argdiffs are provided for
retained applications whose arguments may have changed. If a retained
application does not appear, then its argdiff if `noargdiff`.
"""
struct MapCustomArgDiff{T}
    retained_argdiffs::Dict{Int,T} 
end

# the kernel function must accept noargdiff and unknownargdiff as argdiffs

export MapCustomArgDiff


###########
# retdiff #
###########

# may return: NoRetDiff, or VectorCustomRetDiff


###############################
# generator interface methods #
###############################

include("assess.jl")
include("propose.jl")
include("initialize.jl")
include("generic_update.jl")
include("force_update.jl")
include("fix_update.jl")
include("free_update.jl")
include("extend.jl")
include("backprop.jl")
