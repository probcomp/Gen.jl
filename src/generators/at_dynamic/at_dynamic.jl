########################
# at_dynamic generator # 
########################

struct AtDynamicTrace{T,U,V}
    call::CallRecord{T}
    kernel_trace::U
    index::V
end

struct AtDynamicChoices{T,U,V}
    trace::AtDynamicTrace{T,U,V}
end

get_call_record(trace::AtDynamicTrace) = trace.call
has_choices(trace::AtDynamicTrace) = has_choices(trace.kernel_trace)
get_choices(trace::AtDynamicTrace) = AtDynamicChoices(trace)
Base.isempty(choices::AtDynamicChoices) = !has_choices(choices.trace.kernel_trace)
get_address_schema(::Type{AtDynamicChoices}) = SingleDynamicKeyAddressSchema()
has_internal_node(choices::AtDynamicChoices, addr) = false
function has_internal_node(choices::AtDynamicChoices{T,U,V}, addr::V) where {T,U,V}
    choices.trace.index == addr
end
function has_internal_node(choices::AtDynamicChoices{T,U,V}, addr::Pair{V,W}) where {T,U,V,W}
    (first, rest) = addr
    if choices.trace.index == addr
        subchoices = get_choices(choices.trace.kernel_trace)
        has_internal_node(subchoices, rest)
    else
        throw(KeyError(first))
    end
end

function get_internal_nodes(choices::AtDynamicChoices)
    node = (choices.trace.index, get_choices(choices.trace.kernel_trace))
    (node,)
end

function get_leaf_nodes(choices::AtDynamicChoices)
    # TODO what about wrapping a distribuion, instead of a a generator (cf Plate)
    ()
end

Base.haskey(choices::AtDynamicChoices, addr) = has_leaf_node(choices, addr)
Base.getindex(choices::AtDynamicChoices, addr) = get_leaf_node(choices, addr)


"""
Generator that applies another generator at a specific, dynamically determined, address.
"""
struct AtDynamic{T,U,V} <: Generator{T, AtDynamicTrace{T,U,V}}
    kernel::Generator{T,U}
end

function at_dynamic(kernel::Generator{T,U}, ::Type{V}) where {T,U,V}
    AtDynamic{T,U,V}(kernel)
end

function get_static_argument_types(gen::AtDynamic{T,U,V}) where {T,U,V}
    [:($V), Expr(:curly, :Tuple, get_static_argument_types(gen.kernel)...)]
end

include("simulate.jl")
include("assess.jl")

export at_dynamic
