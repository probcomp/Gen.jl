using FunctionalCollections: PersistentVector

"""

U is the type of the subtrace, R is the return value type for the kernel
"""
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
get_choices(trace::VectorDistTrace) = VectorDistTraceChoiceTrie(trace)

struct VectorDistTraceChoiceTrie
    trace::VectorDistTrace
end

Base.isempty(choices::VectorDistTraceChoiceTrie) = !has_choices(choices.trace)
get_address_schema(::Type{VectorDistTraceChoiceTrie}) = VectorAddressSchema()
has_internal_node(choices::VectorDistTraceChoiceTrie, addr) = false
has_leaf_node(choices::VectorDistTraceChoiceTrie, addr) = false

function get_leaf_node(choices::VectorDistTraceChoiceTrie, addr::Int)
    choices.trace.values[addr]
end

get_internal_nodes(choices::VectorDistTraceChoiceTrie) = ()
get_leaf_nodes(choices::VectorDistTraceChoiceTrie) = choices.values
Base.haskey(choices::VectorDistTraceChoiceTrie, addr) = has_leaf_node(choices, addr)
Base.getindex(choices::VectorDistTraceChoiceTrie, addr) = get_leaf_node(choices, addr)
