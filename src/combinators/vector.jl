using FunctionalCollections: PersistentVector, assoc, push, pop

###########################################
# trace used by vector-shaped combinators #
###########################################

"""

U is the type of the subtrace, R is the return value type for the kernel
"""
struct VectorTrace{GenFnType,T,U} <: Trace
    gen_fn::GenerativeFunction
    subtraces::PersistentVector{U}
    retval::PersistentVector{T}
    args::Tuple

    # number of active subtraces (may be less than length of subtraces)
    len::Int

    # number of active subtraces that are nonempty (used for has_choices)
    num_nonempty::Int

    score::Float64
    noise::Float64
end

function VectorTrace{GenFnType,T,U}(gen_fn::GenerativeFunction,
                                    subtraces::PersistentVector{U},
                                    retval::PersistentVector{T},
                                    args::Tuple, score::Float64, noise::Float64,
                                    len::Int, num_nonempty::Int) where {GenFnType,T,U}
    @assert length(subtraces) == length(retval)
    @assert length(subtraces) == len
    @assert num_nonempty >= 0
    VectorTrace{GenFnType,T,U}(gen_fn, subtraces, retval, args, len,
        num_nonempty, score, noise)
end

function VectorTrace{GenFnType,T,U}(gen_fn::GenerativeFunction, args::Tuple) where {GenFnType,T,U}
    subtraces = PersistentVector{U}()
    retvals = PersistentVector{T}()
    VectorTrace{GenFnType,T,U}(gen_fn, subtraces, retvals, args, 0, 0, 0., 0.)
end

# trace API

get_choices(trace::VectorTrace) = VectorTraceChoiceMap(trace)
get_retval(trace::VectorTrace) = trace.retval
get_args(trace::VectorTrace) = trace.args
get_score(trace::VectorTrace) = trace.score
get_gen_fn(trace::VectorTrace) = trace.gen_fn

function project(trace::VectorTrace, selection::AddressSet)
    if !isempty(get_leaf_nodes(selection))
        error("An entire sub-assignment was selected at key $key")
    end
    weight = 0.
    for key=1:trace.len
        if has_internal_node(selection, key)
            subselection = get_internal_node(selection, key)
        else
            subselection = EmptyAddressSet()
        end
        weight += project(trace.subtraces[key], subselection)
    end
    weight
end

struct VectorTraceChoiceMap <: ChoiceMap
    trace::VectorTrace
end

Base.isempty(assignment::VectorTraceChoiceMap) = assignment.trace.num_nonempty == 0
get_address_schema(::Type{VectorTraceChoiceMap}) = VectorAddressSchema()

function get_submap(choices::VectorTraceChoiceMap, addr::Int)
    if addr <= choices.trace.len
        get_choices(choices.trace.subtraces[addr])
    else
        EmptyChoiceMap()
    end
end

function get_submaps_shallow(choices::VectorTraceChoiceMap)
    ((i, get_choices(choices.trace.subtraces[i])) for i=1:choices.trace.len)
end

get_submap(choices::VectorTraceChoiceMap, addr::Pair) = _get_submap(choices, addr)
get_value(choices::VectorTraceChoiceMap, addr::Pair) = _get_value(choices, addr)
has_value(choices::VectorTraceChoiceMap, addr::Pair) = _has_value(choices, addr)
get_values_shallow(::VectorTraceChoiceMap) = ()


############################################
# code shared by vector-shaped combinators #
############################################

function get_retained_and_constrained(constraints::ChoiceMap, prev_length::Int, new_length::Int)
    keys = Set{Int}()
    for (key::Int, _) in get_submaps_shallow(constraints)
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
    retdiff = VectorCustomRetDiff(retained_retdiffs:Dict{Int,Any})

Construct a retdiff that provides retdiff information about some elements of the returned vector.

    retdiff[i]

Return the retdiff value for the `i`th element of the vector.

    haskey(retdiff, i::Int)

Return true if there is a retdiff value for the `i`th element of the vector, or false if there was no difference in this element.

    keys(retdiff)

Return an iterator over the elements with retdiff values.
"""
struct VectorCustomRetDiff
    retained_retdiffs::Dict{Int,Any}
end

VectorCustomRetDiff() = VectorCustomRetDiff(Dict{Int,Any}())

isnodiff(::VectorCustomRetDiff) = false

function Base.getindex(retdiff::VectorCustomRetDiff, i::Int)
    retdiff.retained_retdiffs[i]
end

function Base.haskey(retdiff::VectorCustomRetDiff, i::Int)
    haskey(retdiff.retained_retdiffs, i)
end

Base.keys(retdiff::VectorCustomRetDiff) = keys(retdiff.retained_retdiffs)

export VectorCustomRetDiff

function vector_compute_retdiff(isdiff_retdiffs::Dict{Int,Any}, new_length::Int, prev_length::Int)
    if new_length == prev_length && length(isdiff_retdiffs) == 0
        NoRetDiff()
    else
        VectorCustomRetDiff(isdiff_retdiffs)
    end
end

function vector_update_delete(new_length::Int, prev_length::Int,
                                 prev_trace::VectorTrace)
    num_nonempty = prev_trace.num_nonempty
    discard = choicemap()
    score_decrement = 0.
    noise_decrement = 0.
    for key=new_length+1:prev_length
        subtrace = prev_trace.subtraces[key]
        score_decrement += get_score(subtrace)
        noise_decrement += project(subtrace, EmptyAddressSet())
        if !isempty(get_choices(subtrace))
            num_nonempty -= 1
        end
        @assert num_nonempty >= 0
        set_submap!(discard, key, get_choices(subtrace))
    end
    return (discard, num_nonempty, score_decrement, noise_decrement)
end

function vector_regenerate_delete(new_length::Int, prev_length::Int,
                                  prev_trace::VectorTrace)
    num_nonempty = prev_trace.num_nonempty
    score_decrement = 0.
    noise_decrement = 0.
    for key=new_length+1:prev_length
        subtrace = prev_trace.subtraces[key]
        score_decrement += get_score(subtrace)
        noise_decrement += project(subtrace, EmptyAddressSet())
        if !isempty(get_choices(subtrace))
            num_nonempty -= 1
        end
        @assert num_nonempty >= 0
    end
    return (num_nonempty, score_decrement, noise_decrement)
end

function vector_remove_deleted_applications(subtraces, retval, prev_length, new_length)
    for i=new_length+1:prev_length
        subtraces = pop(subtraces)
        retval = pop(retval)
    end
    (subtraces, retval)
end
