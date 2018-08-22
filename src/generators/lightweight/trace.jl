using FunctionalCollections: PersistentHashMap, assoc


#############################
# generative function trace #
#############################

mutable struct GFTrace
    call::Union{CallRecord,Nothing}
    primitive_calls::PersistentHashMap{Any,CallRecord}

    # values can be a GFTrace, or Any (foreign trace)
    subtraces::PersistentHashMap{Any,Any} 
    has_choices::Bool
end

function GFTrace()
    primitive_calls = PersistentHashMap{Any,CallRecord}()
    subtraces = PersistentHashMap{Any,Any}()
    call = nothing
    GFTrace(call, primitive_calls, subtraces, false)
end

get_call_record(trace::GFTrace) = trace.call
has_choices(trace::GFTrace) = trace.has_choices

function has_primitive_call(trace::GFTrace, addr)
    haskey(trace.primitive_calls, addr)
end

function has_primitive_call(trace::GFTrace, addr::Pair)
    (first, rest) = addr
    subtrace::GFTrace = get(trace.subtraces, first)
    has_primitive_call(subtrace, rest)
end

function has_subtrace(trace::GFTrace, addr)
    haskey(trace.subtraces, addr)
end

function has_subtrace(trace::GFTrace, addr::Pair)
    (first, rest) = addr
    if !haskey(trace.subtraces, first)
        return false
    end
    subtrace::GFTrace = get(trace.subtraces, first)
    has_subtrace(subtrace, rest)
end

function get_primitive_call(trace::GFTrace, addr)
    get(trace.primitive_calls, addr)
end

function get_primitive_call(trace::GFTrace, addr::Pair)
    (first, rest) = addr
    subtrace::GFTrace = get(trace.subtraces, first)
    get_primitive_call(subtrace, rest)
end

function assoc_primitive_call(trace::GFTrace, addr, call::CallRecord)
    primitive_calls = assoc(trace.primitive_calls, addr, call)
    GFTrace(trace.call, primitive_calls, trace.subtraces, true)
end

function assoc_primitive_call(trace::GFTrace, addr::Pair, call::CallRecord)
    (first, rest) = addr
    local node::GFTrace
    local subtraces::PersistentHashMap{Any,Any}
    if !haskey(trace.subtraces, first)
        node = GFTrace()
        subtraces = assoc(trace.subtraces, first, node)
    else
        node = get(trace.subtraces, first)
        subtraces = trace.subtraces
    end
    node = assoc_primitive_call(node, rest, call)
    subtraces = assoc(subtraces, first, node)
    GFTrace(trace.call, trace.primitive_calls, subtraces, true)
end

function get_subtrace(trace::GFTrace, addr)
    get(trace.subtraces, addr)
end

function get_subtrace(trace::GFTrace, addr::Pair)
    (first, rest) = addr
    subtrace::GFTrace = get(trace.subtraces, first)
    get_subtrace(subtrace, rest)
end

function assoc_subtrace(trace::GFTrace, addr, subtrace)
    subtraces = assoc(trace.subtraces, addr, subtrace)
    GFTrace(trace.call, trace.primitive_calls, subtraces,
            has_choices(subtrace) || trace.has_choices)
end

function assoc_subtrace(trace::GFTrace, addr::Pair, subtrace)
    (first, rest) = addr
    local node::GFTrace
    local subtraces::PersistentHashMap{Any,Any}
    if !haskey(trace.subtraces, first)
        node = GFTrace()
        subtraces = assoc(trace.subtraces, first, node)
    else
        node = get(trace.subtraces, first)
        subtraces = trace.subtraces
    end
    node = assoc_subtrace(node, rest, subtrace)
    subtraces = assoc(subtraces, first, node)
    GFTrace(trace.call, trace.primitive_calls, subtraces,
            has_choices(subtrace) || trace.has_choices)
end


##############################################
# choice trie for generative function traces #
##############################################

struct GFTraceChoices <: ChoiceTrie
    trace::GFTrace
end

get_choices(trace::GFTrace) = GFTraceChoices(trace)
get_address_schema(::Type{GFTraceChoices}) = DynamicAddressSchema()
Base.isempty(choices::GFTraceChoices) = !has_choices(choices.trace)

function has_internal_node(choices::GFTraceChoices, addr::Pair)
    (first, rest) = addr
    subtrace = choices.trace.subtraces[first]
    has_internal_node(get_choices(subtrace), rest)
end

function has_internal_node(choices::GFTraceChoices, addr)
    haskey(choices.trace.subtraces, addr) && has_choices(choices.trace.subtraces[addr])
end

function get_internal_node(choices::GFTraceChoices, addr::Pair)
    (first, rest) = addr
    subtrace = choices.trace.subtraces[first]
    if !has_choices(subtrace)
        throw(KeyError(addr))
    end
    get_internal_node(get_choices(subtrace), rest)
end

function get_internal_node(choices::GFTraceChoices, addr)
    subtrace = choices.trace.subtraces[addr]
    if !has_choices(subtrace)
        throw(KeyError(addr))
    end
    get_choices(subtrace)
end

function has_leaf_node(choices::GFTraceChoices, addr::Pair)
    (first, rest) = addr
    if !haskey(choices.trace.subtraces, first)
        return false
    end
    sub_choices = get_choices(choices.trace.subtraces[first])
    has_leaf_node(sub_choices, rest)
end

function has_leaf_node(choices::GFTraceChoices, addr)
    haskey(choices.trace.primitive_calls, addr)
end

function get_leaf_node(choices::GFTraceChoices, addr::Pair)
    (first, rest) = addr
    sub_choices = get_choices(choices.trace.subtraces[first])
    get_leaf_node(sub_choices, rest)
end

function get_leaf_node(choices::GFTraceChoices, addr)
    choices.trace.primitive_calls[addr].retval
end

function get_leaf_nodes(choices::GFTraceChoices)
    ((key, call.retval) for (key, call) in choices.trace.primitive_calls)
end

function get_internal_nodes(choices::GFTraceChoices)
    ((key, get_choices(subtrace)) for (key, subtrace) in choices.trace.subtraces
     if has_choices(subtrace))
end
