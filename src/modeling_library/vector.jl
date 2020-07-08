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

@inline get_choices(trace::VectorTrace) = VectorTraceChoiceMap(trace)
@inline get_retval(trace::VectorTrace) = trace.retval
@inline get_args(trace::VectorTrace) = trace.args
@inline get_score(trace::VectorTrace) = trace.score
@inline get_gen_fn(trace::VectorTrace) = trace.gen_fn

@inline function Base.getindex(trace::VectorTrace{GenFnType, T, U}, addr::Pair) where {GenFnType, T, U}
    (first, rest) = addr
    subtrace = trace.subtraces[first]
    subtrace[rest]
end

@inline function Base.getindex(trace::VectorTrace, addr)
    # we expose the return values in the auxiliary state
    trace.retval[addr]
end


function project(trace::VectorTrace, selection::Selection)
    weight = 0.
    for key=1:trace.len
        subselection = get_subselection(selection, key)
        weight += project(trace.subtraces[key], subselection)
    end
    weight
end
project(trace::VectorTrace, ::EmptySelection) = trace.noise

struct VectorTraceChoiceMap{GenFnType, T, U} <: AddressTree{Value}
    trace::VectorTrace{GenFnType, T, U}
end

@inline Base.isempty(assignment::VectorTraceChoiceMap) = assignment.trace.num_nonempty == 0
@inline get_address_schema(::Type{VectorTraceChoiceMap}) = VectorAddressSchema()

@inline get_subtree(choices::VectorTraceChoiceMap, addr::Pair) = _get_subtree(choices, addr)
@inline function get_subtree(choices::VectorTraceChoiceMap, addr::Int)
    if addr <= choices.trace.len
        get_choices(choices.trace.subtraces[addr])
    else
        EmptyChoiceMap()
    end
end
# keys which are not ints have no sub-choicemap
@inline get_subtree(choices::VectorTraceChoiceMap, addr) = EmptyChoiceMap()

@inline function get_subtrees_shallow(choices::VectorTraceChoiceMap)
    ((i, get_choices(choices.trace.subtraces[i])) for i=1:choices.trace.len)
end

############################################
# code shared by vector-shaped combinators #
############################################

function get_retained_and_specd(spec::UpdateSpec, prev_length::Int, new_length::Int)
    keys = Set{Int}()
    for (key::Int, subspec) in get_subtrees_shallow(spec)
        isempty(subspec) && continue;
        if key > 0 && key <= new_length
            push!(keys, key)
        else
            error("Update spec included address which does not exist: $key")
        end
    end
    keys 
end

function get_retained_and_selected(selection::EmptySelection, prev_length::Int, new_length::Int)
    Set{Int}()
end

function get_retained_and_selected(selection::Selection, prev_length::Int, new_length::Int)
    keys = Set{Int}()
    for (key::Int, _) in get_subselections(selection)
        if key > 0 && key <= new_length
            push!(keys, key)
        else
            error("Selected address does not exist: $key")
        end
    end
    keys 
end

function get_retained_and_selected(selection::AllSelection, prev_length::Int, new_length::Int)
    Set{Int}(1:min(prev_length, new_length))
end

function vector_compute_retdiff(updated_retdiffs::Dict{Int,Diff}, new_length::Int, prev_length::Int)
    if new_length == prev_length && length(updated_retdiffs) == 0
        NoChange()
    else
        VectorDiff(new_length, prev_length, updated_retdiffs)
    end
end

function vector_update_delete(new_length::Int, prev_length::Int,
                                 prev_trace::VectorTrace, externally_constrained_addrs::Selection)
    num_nonempty = prev_trace.num_nonempty
    discard = choicemap()
    score_decrement = 0.
    noise_decrement = 0.
    deletion_weight = 0.
    for key=new_length+1:prev_length
        subtrace = prev_trace.subtraces[key]
        score_change = get_score(subtrace)
        noise_change = project(subtrace, EmptySelection())
       
        score_decrement += score_change
        noise_decrement += noise_change

        ext_const = get_subselection(externally_constrained_addrs, key)
        if isempty(ext_const)
            deletion_weight += noise_change
        elseif ext_const === AllSelection()
            deletion_weight += score_change
        else
            deletion_weight += project(subtrace, addrs(get_selected(get_choices(subtrace), ext_const)))
        end

        if !isempty(get_choices(subtrace))
            num_nonempty -= 1
        end
        @assert num_nonempty >= 0
        set_submap!(discard, key, get_choices(subtrace))
    end
    return (discard, num_nonempty, score_decrement, noise_decrement, deletion_weight)
end

function vector_remove_deleted_applications(subtraces, retval, prev_length, new_length)
    for i=new_length+1:prev_length
        subtraces = pop(subtraces)
        retval = pop(retval)
    end
    (subtraces, retval)
end
