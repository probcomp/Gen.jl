using FunctionalCollections: PersistentVector

#######################################
# trace for vector of generator calls #
#######################################

"""

U is the type of the subtrace, R is the return value type for the kernel
"""
struct VectorTrace{T,U}
    subtraces::PersistentVector{U}
    call::CallRecord{PersistentVector{T}}
    is_empty::Bool
end

function VectorTrace{T,U}() where {T,U}
    VectorTrace{T,U}(PersistentVector{U}(), PersistentVector{T}(), true)
end

function get_subtrace(trace::VectorTrace{T,U}, i::Int) where {T,U}
    get(trace.subtraces, i)
end

# TODO need to manage is_empty

# trace API

get_call_record(trace::VectorTrace) = trace.call
has_choices(trace::VectorTrace) = !trace.is_empty
get_assignment(trace::VectorTrace) = VectorTraceAssignment(trace)

struct VectorTraceAssignment <: Assignment
    trace::VectorTrace
end

Base.isempty(assignment::VectorTraceAssignment) = assignment.trace.is_empty
get_address_schema(::Type{VectorTraceAssignment}) = VectorAddressSchema()
has_internal_node(assignment::VectorTraceAssignment, addr) = false

function has_internal_node(assignment::VectorTraceAssignment, addr::Int)
    n = length(assignment.trace.subtraces)
    addr >= 1 && addr <= n
end

function has_internal_node(assignment::VectorTraceAssignment, addr::Pair)
    (first, rest) = addr
    sub_assignment = get_assignment(assignment.trace.subtraces[first])
    has_internal_node(sub_assignment, rest)
end

function get_internal_node(assignment::VectorTraceAssignment, addr::Int)
    get_assignment(assignment.trace.subtraces[addr])
end

function get_internal_node(assignment::VectorTraceAssignment, addr::Pair)
    (first, rest) = addr
    sub_assignment = get_assignment(assignment.trace.subtraces[first])
    get_internal_node(sub_assignment, rest)
end

has_leaf_node(assignment::VectorTraceAssignment, addr) = false

function has_leaf_node(assignment::VectorTraceAssignment, addr::Pair)
    (first, rest) = addr
    sub_assignment = get_assignment(assignment.trace.subtraces[first])
    has_leaf_node(sub_assignment, rest)
end

function get_leaf_node(assignment::VectorTraceAssignment, addr::Pair)
    (first, rest) = addr
    sub_assignment = get_assignment(assignment.trace.subtraces[first])
    get_leaf_node(sub_assignment, rest)
end

function get_internal_nodes(assignment::VectorTraceAssignment)
    [(i, get_assignment(assignment.trace.subtraces[i])) for i=1:length(assignment.trace.subtraces)]
end

get_leaf_nodes(assignment::VectorTraceAssignment) = []


##########################################
# trace for vector of distribution calls #
##########################################

struct VectorDistTrace{T}
    values::PersistentVector{T}
    call::CallRecord{PersistentVector{T}}
end

function VectorDistTrace{T}() where {T}
    VectorDistTrace{T}(PersistentVector{T}())
end

# trace API

get_call_record(trace::VectorDistTrace) = trace.call
has_choices(trace::VectorDistTrace) = length(trace.values) > 0
get_assignment(trace::VectorDistTrace) = VectorDistTraceAssignment(trace)

struct VectorDistTraceAssignment <: Assignment
    trace::VectorDistTrace
end

Base.isempty(assignment::VectorDistTraceAssignment) = !has_choices(assignment.trace)
get_address_schema(::Type{VectorDistTraceAssignment}) = VectorAddressSchema()
has_internal_node(assignment::VectorDistTraceAssignment, addr) = false
has_leaf_node(assignment::VectorDistTraceAssignment, addr) = false

function get_leaf_node(assignment::VectorDistTraceAssignment, addr::Int)
    assignment.trace.values[addr]
end

get_internal_nodes(assignment::VectorDistTraceAssignment) = ()
get_leaf_nodes(assignment::VectorDistTraceAssignment) = assignment.values
