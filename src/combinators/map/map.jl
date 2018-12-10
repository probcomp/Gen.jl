###################
# map generator # 
###################

struct MapType end # used for type dispatch on the VectorTrace type (e.g. we will also have a UnfoldType)

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

function get_args_for_key(args::Tuple, key::Int)
    map((arg) -> arg[key], args)
end

function get_prev_and_new_lengths(args::Tuple, prev_trace)
    new_length = length(args[1])
    prev_args = get_args(prev_trace)
    prev_length = length(prev_args[1])
    (new_length, prev_length)
end

"""
Collect constraints indexed by the integer key; check validity of addresses.
"""
function collect_map_constraints(constraints::Assignment, len::Int)
    if length(collect(get_values_shallow(constraints))) > 0
        bad_addr = first(get_values_shallow(constraints))[1]
        error("Constrained address does not exist: $bad_addr")
    end
    subassmts = Dict{Int,Any}()
    for (key::Int, subassmt) in get_subassmts_shallow(constraints)
        if key <= len
            subassmts[key] =  subassmt
        else
            error("Constrained address prefix does not exist: $key")
        end
    end
    subassmts
end


function get_retained_and_constrained(constraints::Assignment, prev_length::Int, new_length::Int)
    keys = Set{Int}()
    for (key::Int, _) in get_subassmts_shallow(constraints)
        if key > 0 && key <= new_length
            push!(keys, key)
        else
            error("Constraint namespace does not exist: $key")
        end
    end
    keys 
end

function get_retained_and_selected(selection::AddressSet, prev_length::Int, new_length::Int)
    keys = Set{Int}()
    for (key::Int, _) in get_internal_nodes(selection)
        if key > 0 && key <= new_length
            push!(keys, key)
        else
            error("Selection namespace does not exist: $key")
        end
    end
    keys 
end


"""
Collect selections indexed by the integer key; check validity of addresses.
"""
function collect_map_selections(selection::AddressSet, prev_length::Int, new_length::Int)
    if length(get_leaf_nodes(selection)) > 0
        bad_addr = first(get_leaf_nodes(selection))
        error("Selected address that does not exist: $bad_addr")
    end
    subselections = Dict{Int, Any}()
    for (key::Int, node) in get_internal_nodes(selection)
        if key <= prev_length && key <= new_length
            subselections[key] = node
        else
            error("Cannot select addresses under namespace: $key")
        end
    end
    subselections
end


function compute_retdiff(isdiff_retdiffs::Dict{Int,Any}, new_length::Int, prev_length::Int)
    if new_length == prev_length && length(isdiff_retdiffs) == 0
        MapNoRetDiff()
    else
        MapCustomRetDiff(isdiff_retdiffs)
    end
end

function map_force_update_delete(new_length::Int, prev_length::Int,
                                 prev_trace::VectorTrace)
    num_nonempty = prev_trace.num_nonempty
    discard = DynamicAssignment()
    score_decrement = 0.
    noise_decrement = 0.
    for key=new_length+1:prev_length
        subtrace = prev_trace.subtraces[key]
        score_decrement += get_score(subtrace)
        noise_decrement += project(subtrace, EmptyAddressSet())
        if !isempty(get_assignment(subtrace))
            num_nonempty -= 1
        end
        @assert num_nonempty >= 0
        set_subassmt!(discard, key, get_assignment(subtrace))
    end
    return (discard, num_nonempty, score_decrement, noise_decrement)
end

function map_fix_free_update_delete(new_length::Int, prev_length::Int,
                                    prev_trace::VectorTrace)
    num_nonempty = prev_trace.num_nonempty
    score_decrement = 0.
    noise_decrement = 0.
    for key=new_length+1:prev_length
        subtrace = prev_trace.subtraces[key]
        score_decrement += get_score(subtrace)
        noise_decrement += project(subtrace, EmptyAddressSet())
        if !isempty(get_assignment(subtrace))
            num_nonempty -= 1
        end
        @assert num_nonempty >= 0
    end
    return (num_nonempty, score_decrement, noise_decrement)
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


###########
# retdiff #
###########

"""
The return value of the Map has not changed.
"""
struct MapNoRetDiff end
isnodiff(::MapNoRetDiff) = true

"""
The number of applications may have changed. retdiff values are provided for
retained applications for which isnodiff() = false.
"""
struct MapCustomRetDiff
    retained_retdiffs::Dict{Int,Any}
end
isnodiff(::MapCustomRetDiff) = false



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
#include("extend.jl")
#include("backprop_trace.jl")
