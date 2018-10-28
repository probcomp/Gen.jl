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

plate(kernel::Generator{T,U}) where {T,U} = Plate{T,U}(kernel)

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

"""
Collect constraints indexed by the integer key; check validity of addresses.
"""
function collect_plate_constraints(constraints::Assignment, len::Int)
    if length(get_leaf_nodes(constraints)) > 0
        bad_addr = first(get_leaf_nodes(constraints))[1]
        error("Constrained address that does not exist: $bad_addr")
    end
    nodes = Dict{Int,Any}()
    for (key::Int, node) in get_internal_nodes(constraints)
        if key <= len
            nodes[key] = node
        else
            error("Address key does not exist: $key")
        end
    end
    return nodes
end

"""
Collect constraints indexed by the integer key; check validity of addresses.
"""
function collect_plate_constraints(constraints::Assignment, prev_length::Int, new_length::Int)
    nodes = Dict{Int,Any}()
    retained_constrained = Set{Int}()
    for (key::Int, node) in get_internal_nodes(constraints)
        if key <= prev_length && key <= new_length
            nodes[key] = node
            push!(retained_constrained, key)
        elseif key <= new_length
            nodes[key] = node
        else
            error("Address key does not exist: $key")
        end
    end
    return (nodes, retained_constrained)
end

function compute_retdiff(isdiff_retdiffs::Dict{Int,Any}, new_length::Int, prev_length::Int)
    if new_length == prev_length && length(isdiff_retdiffs) == 0
        PlateNoRetDiff()
    else
        PlateCustomRetDiff(isdiff_retdiffs)
    end
end

function discard_deleted_applications(new_length::Int, prev_length::Int,
                                      prev_trace::VectorTrace)
    num_has_choices = prev_trace.num_has_choices
    discard = DynamicAssignment()
    for key=new_length+1:prev_length
        subtrace = prev_trace.subtraces[key]
        if has_choices(subtrace)
            num_has_choices -= 1
        end
        @assert num_has_choices >= 0
        set_internal_node!(discard, key, get_assignment(subtrace))
    end
    return (discard, num_has_choices)
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

# the kernel function must accept noargdiff and unknownargdiff as argdiffs


###########
# retdiff #
###########

"""
The return value of the plate has not changed.
"""
struct PlateNoRetDiff end
isnodiff(::PlateNoRetDiff) = true

"""
The number of applications may have changed. retdiff values are provided for
retained applications for which isnodiff() = false.
"""
struct PlateCustomRetDiff
    retained_retdiffs::Dict{Int,Any}
end
isnodiff(::PlateCustomRetDiff) = false



###############################
# generator interface methods #
###############################

include("assess.jl")
include("generate.jl")
include("update.jl")
include("project.jl")
include("backprop_trace.jl")
