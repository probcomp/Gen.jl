########################
# at_dynamic generator # 
########################

struct AtDynamicTrace{T,U,K}
    call::CallRecord{T}
    kernel_trace::U
    key::K
end

struct AtDynamicAssignment{T,U,K} <: Assignment
    trace::AtDynamicTrace{T,U,K}
end

get_call_record(trace::AtDynamicTrace) = trace.call
has_choices(trace::AtDynamicTrace) = has_choices(trace.kernel_trace)
get_assignment(trace::AtDynamicTrace) = AtDynamicAssignment(trace)
Base.isempty(assignment::AtDynamicAssignment) = !has_choices(assignment.trace.kernel_trace)
get_address_schema(::Type{AtDynamicAssignment}) = SingleDynamicKeyAddressSchema()
function has_internal_node(assignment::AtDynamicAssignment{T,U,K}, addr::K) where {T,U,K}
    assignment.trace.key == addr
end
function has_internal_node(assignment::AtDynamicAssignment{T,U,K}, addr::Pair{K,W}) where {T,U,K,W}
    (first, rest) = addr
    if assignment.trace.key == first
        sub_assignment = get_assignment(assignment.trace.kernel_trace)
        has_internal_node(sub_assignment, rest)
    else
        throw(KeyError(first))
    end
end
function get_internal_node(assignment::AtDynamicAssignment{T,U,K}, addr::K) where {T,U,K}
    if assignment.trace.key == addr
        get_assignment(assignment.trace.kernel_trace)
    else
        throw(KeyError(addr))
    end
end
function get_internal_node(assignment::AtDynamicAssignment{T,U,K}, addr::Pair{K,W}) where {T,U,K,W}
    (first, rest) = addr
    if assignment.trace.key == first
        sub_assignment = get_assignment(assignment.trace.kernel_trace)
        get_internal_node(sub_assignment, rest)
    else
        throw(KeyError(first))
    end
end


function get_internal_nodes(assignment::AtDynamicAssignment)
    node = (assignment.trace.key, get_assignment(assignment.trace.kernel_trace))
    (node,)
end

function get_leaf_nodes(assignment::AtDynamicAssignment)
    # TODO what about wrapping a distribuion, instead of a a generator (cf Plate)
    ()
end


"""
Generator that applies another generator at a specific, dynamically determined, address.
"""
struct AtDynamic{T,U,K} <: Generator{T, AtDynamicTrace{T,U,K}}
    kernel::Generator{T,U}
end

function at_dynamic(kernel::Generator{T,U}, ::Type{K}) where {T,U,K}
    AtDynamic{T,U,K}(kernel)
end

function get_static_argument_types(gen::AtDynamic{T,U,K}) where {T,U,K}
    [K, Tuple{get_static_argument_types(gen.kernel)...}]
end

include("simulate.jl")
include("assess.jl")

export at_dynamic
