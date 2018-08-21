using FunctionalCollections: PersistentVector

# Q: TODO can we infer the trace implementation type U?
# A: Yes, if the generator also declares its trace type (which it should)
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

# TODO perhaps provide some abstract here for generator method implementations to use

# TODO need to manage is_empty

# trace API

get_call_record(trace::VectorTrace) = trace.call

has_choices(trace::VectorTrace) = !trace.is_empty

get_choices(trace::VectorTrace) = VectorTraceChoiceTrie(trace)

struct VectorTraceChoiceTrie
    trace::VectorTrace
end

Base.isempty(choices::VectorTraceChoiceTrie) = choices.trace.is_empty

get_address_schema(::Type{VectorTraceChoiceTrie}) = VectorAddressSchema()

has_internal_node(choices::VectorTraceChoiceTrie, addr) = false

function has_internal_node(choices::VectorTraceChoiceTrie, addr::Int)
    n = length(choices.trace.subtraces)
    addr >= 1 && addr <= n
end

function has_internal_node(choices::VectorTraceChoiceTrie, addr::Pair)
    (first, rest) = addr
    subchoices = get_choices(choices.trace.subtraces[first])
    has_internal_node(subchoices, rest)
end

function get_internal_node(choices::VectorTraceChoiceTrie, addr::Int)
    get_choices(choices.trace.subtraces[addr])
end

function get_internal_node(choices::VectorTraceChoiceTrie, addr::Pair)
    (first, rest) = addr
    subchoices = get_choices(choices.trace.subtraces[first])
    get_internal_node(subchoices, rest)
end

has_leaf_node(choices::VectorTraceChoiceTrie, addr) = false

function has_leaf_node(choices::VectorTraceChoiceTrie, addr::Pair)
    (first, rest) = addr
    subchoices = get_choices(choices.trace.subtraces[first])
    has_leaf_node(subchoices, rest)
end

function get_leaf_node(choices::VectorTraceChoiceTrie, addr::Pair)
    (first, rest) = addr
    subchoices = get_choices(choices.trace.subtraces[first])
    get_leaf_node(subchoices, rest)
end

function get_internal_nodes(choices::VectorTraceChoiceTrie)
    [(i, get_choices(choices.trace.subtraces[i])) for i=1:length(choices.trace.subtraces)]
end

# TODO handle the case when the kernel is a disribution!
get_leaf_nodes(choices::VectorTraceChoiceTrie) = []

Base.haskey(choices::VectorTraceChoiceTrie, addr) = has_leaf_node(choices, addr)
Base.getindex(choices::VectorTraceChoiceTrie, addr) = get_leaf_node(choices, addr)
