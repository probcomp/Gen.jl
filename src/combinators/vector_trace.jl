using FunctionalCollections: PersistentVector, assoc, push, pop

#################################################
# trace for vector of generative function calls #
#################################################

"""

U is the type of the subtrace, R is the return value type for the kernel
"""
struct VectorTrace{GenFnType,T,U}
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
    #@assert length(subtraces) >= len
    #@assert length(subtraces) >= len
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

get_assignment(trace::VectorTrace) = VectorTraceAssignment(trace)
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

struct VectorTraceAssignment <: Assignment
    trace::VectorTrace
end

Base.isempty(assignment::VectorTraceAssignment) = assignment.trace.num_nonempty == 0
get_address_schema(::Type{VectorTraceAssignment}) = VectorAddressSchema()

function get_subassmt(assmt::VectorTraceAssignment, addr::Int)
    if addr <= assmt.trace.len
        get_assignment(assmt.trace.subtraces[addr])
    else
        EmptyAssignment()
    end
end

function get_subassmts_shallow(assmt::VectorTraceAssignment)
    ((i, get_assignment(assmt.trace.subtraces[i])) for i=1:assmt.trace.len)
end

get_subassmt(assmt::VectorTraceAssignment, addr::Pair) = _get_subassmt(assmt, addr)
get_value(assmt::VectorTraceAssignment, addr::Pair) = _get_value(assmt, addr)
has_value(assmt::VectorTraceAssignment, addr::Pair) = _has_value(assmt, addr)
get_values_shallow(::VectorTraceAssignment) = ()
