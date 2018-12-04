using FunctionalCollections: PersistentVector

#######################################
# trace for vector of generator calls #
#######################################

# TODO add bounds checking and reference counting

"""

U is the type of the subtrace, R is the return value type for the kernel
"""
struct VectorTrace{T,U}
    subtraces::PersistentVector{U}
    call::CallRecord{PersistentVector{T}}

    # number of active subtraces (may be less than length of subtraces)
    len::Int

    # number of active subtraces that are nonempty (used for has_choices)
    num_has_choices::Int
end

function VectorTrace{T,U}(subtraces::PersistentVector{U},
                          retvals::PersistentVector{T},
                          args::Tuple, score::Float64,
                          len::Int, num_has_choices::Int) where {T,U}
    @assert length(subtraces) == length(retvals)
    @assert length(subtraces) >= len
    @assert num_has_choices >= 0
    call = CallRecord{PersistentVector{T}}(score, retvals, args)
    VectorTrace{T,U}(subtraces, call, len, num_has_choices)
end

function VectorTrace{T,U}(args::Tuple) where {T,U}
    subtraces = PersistentVector{U}()
    retvals = PersistentVector{T}()
    VectorTrace{T,U}(subtraces, retvals, args, 0., 0, 0)
end

# trace API

get_call_record(trace::VectorTrace) = trace.call
has_choices(trace::VectorTrace) = trace.num_has_choices > 0
get_assignment(trace::VectorTrace) = VectorTraceAssignment(trace)

struct VectorTraceAssignment <: Assignment
    trace::VectorTrace
end

Base.isempty(assignment::VectorTraceAssignment) = has_choices(assignment.trace)
get_address_schema(::Type{VectorTraceAssignment}) = VectorAddressSchema()

function has_internal_node(assignment::VectorTraceAssignment, addr::Int)
    addr >= 1 && addr <= assignment.trace.len
end

function get_internal_node(assignment::VectorTraceAssignment, addr::Int)
    if addr <= assignment.trace.len
        get_assignment(assignment.trace.subtraces[addr])
    else
        throw(BoundsError(assignment, addr))
    end
end

function get_internal_nodes(assignment::VectorTraceAssignment)
    ((i, get_assignment(assignment.trace.subtraces[i])) for i=1:assignment.trace.len)
end

has_internal_node(assignment::VectorTraceAssignment, addr::Pair) = _has_internal_node(assignment, addr)
get_internal_node(assignment::VectorTraceAssignment, addr::Pair) = _get_internal_node(assignment, addr)
get_leaf_node(assignment::VectorTraceAssignment, addr::Pair) = _get_leaf_node(assignment, addr)
has_leaf_node(assignment::VectorTraceAssignment, addr::Pair) = _has_leaf_node(assignment, addr)

get_leaf_nodes(assignment::VectorTraceAssignment) = ()


##########################################
# trace for vector of distribution calls #
##########################################

struct VectorDistTrace{T}
    values::PersistentVector{T}
    call::CallRecord{PersistentVector{T}}
    len::Int
end

function VectorDistTrace(values::PersistentVector{T},
                         args::Tuple, score::Float64,
                         len::Int) where {T}
    @assert length(values) >= len
    call = CallRecord{PersistentVector{T}}(score, values, args)
    VectorDistTrace{T}(values, call, len)
end

function VectorDistTrace{T}(args::Tuple) where {T}
    values = PersistentVector{T}()
    VectorDistTrace{T}(values, args, 0., 0)
end

# trace API

get_call_record(trace::VectorDistTrace) = trace.call
has_choices(trace::VectorDistTrace) = trace.len > 0
get_assignment(trace::VectorDistTrace) = VectorDistTraceAssignment(trace)

struct VectorDistTraceAssignment <: Assignment
    trace::VectorDistTrace
end

Base.isempty(assignment::VectorDistTraceAssignment) = !has_choices(assignment.trace)
get_address_schema(::Type{VectorDistTraceAssignment}) = VectorAddressSchema()

function has_leaf_node(assignment::VectorDistTraceAssignment, addr::Int)
    addr >= 1 && addr <= assignment.trace.len
end

function get_leaf_node(assignment::VectorDistTraceAssignment, addr::Int)
    if addr <= assignment.trace.len
        assignment.trace.values[addr]
    else
        throw(BoundsError(assignment, addr))
    end
end

get_internal_nodes(assignment::VectorDistTraceAssignment) = ()

function get_leaf_nodes(assignment::VectorDistTraceAssignment)
    (assignment.values[i] for i=1:assignment.trace.len)
end
