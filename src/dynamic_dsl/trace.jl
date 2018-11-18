using FunctionalCollections: PersistentHashMap, assoc


#############################
# generative function trace #
#############################

mutable struct GFTrace
    call::Union{CallRecord,Nothing}
    primitive_calls::PersistentHashMap{Any,CallRecord}

    # values can be a GFTrace, or Any (foreign trace)
    subtraces::PersistentHashMap{Any,Any} 

    # number of subtraces that have choices
    num_has_choices::Int
end

function GFTrace()
    primitive_calls = PersistentHashMap{Any,CallRecord}()
    subtraces = PersistentHashMap{Any,Any}()
    call = nothing
    GFTrace(call, primitive_calls, subtraces, 0)
end

get_call_record(trace::GFTrace) = trace.call

function has_choices(trace::GFTrace)
    length(trace.primitive_calls) > 0 || trace.num_has_choices > 0
end

function has_primitive_call(trace::GFTrace, addr)
    haskey(trace.primitive_calls, addr)
end

function has_primitive_call(trace::GFTrace, addr::Pair)
    (first, rest) = addr
    if !haskey(trace.subtraces, first)
        return false
    end
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
    GFTrace(trace.call, primitive_calls, trace.subtraces, trace.num_has_choices)
end

function assoc_primitive_call(trace::GFTrace, addr::Pair, call::CallRecord)
    (first, rest) = addr
    local subtrace::GFTrace
    if has_subtrace(trace, first)
        subtrace = get(trace.subtraces, first)
    else
        subtrace = GFTrace()
    end
    subtrace = assoc_primitive_call(subtrace, rest, call)
    assoc_subtrace(trace, first, subtrace)
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
    num_has_choices = trace.num_has_choices
    if has_subtrace(trace, addr)
        prev_subtrace = get_subtrace(trace, addr)
        if has_choices(prev_subtrace) && !has_choices(subtrace)
            num_has_choices -= 1
        elseif !has_choices(prev_subtrace) && has_choices(subtrace)
            num_has_choices += 1
        end
    else
        if has_choices(subtrace)
            num_has_choices += 1
        end
    end
    subtraces = assoc(trace.subtraces, addr, subtrace)
    GFTrace(trace.call, trace.primitive_calls, subtraces, num_has_choices)
end

function assoc_subtrace(trace::GFTrace, addr::Pair, subtrace)
    (first, rest) = addr
    local internal_subtrace::GFTrace
    if has_subtrace(trace, first)
        internal_subtrace = get_subtrace(trace, first)
    else
        internal_subtrace = GFTrace()
    end
    internal_subtrace = assoc_subtrace(internal_subtrace, rest, subtrace)
    assoc_subtrace(trace, first, internal_subtrace)
end


##################################################
# assignment wrapping generative function traces #
##################################################

struct GFTraceAssignment <: Assignment
    trace::GFTrace
end

get_assignment(trace::GFTrace) = GFTraceAssignment(trace)
get_address_schema(::Type{GFTraceAssignment}) = DynamicAddressSchema()
Base.isempty(assignment::GFTraceAssignment) = !has_choices(assignment.trace)

function has_internal_node(assignment::GFTraceAssignment, addr::Pair)
    (first, rest) = addr
    if haskey(assignment.trace.subtraces, first)
        subtrace = assignment.trace.subtraces[first]
        has_internal_node(get_assignment(subtrace), rest)
    else
        false
    end
end

function has_internal_node(assignment::GFTraceAssignment, addr)
    haskey(assignment.trace.subtraces, addr) && has_choices(assignment.trace.subtraces[addr])
end

function get_internal_node(assignment::GFTraceAssignment, addr::Pair)
    (first, rest) = addr
    subtrace = assignment.trace.subtraces[first]
    if !has_choices(subtrace)
        throw(KeyError(addr))
    end
    get_internal_node(get_assignment(subtrace), rest)
end

function get_internal_node(assignment::GFTraceAssignment, addr)
    subtrace = assignment.trace.subtraces[addr]
    if !has_choices(subtrace)
        throw(KeyError(addr))
    end
    get_assignment(subtrace)
end

function has_leaf_node(assignment::GFTraceAssignment, addr::Pair)
    (first, rest) = addr
    if !haskey(assignment.trace.subtraces, first)
        return false
    end
    sub_assignment = get_assignment(assignment.trace.subtraces[first])
    has_leaf_node(sub_assignment, rest)
end

function has_leaf_node(assignment::GFTraceAssignment, addr)
    haskey(assignment.trace.primitive_calls, addr)
end

function get_leaf_node(assignment::GFTraceAssignment, addr::Pair)
    (first, rest) = addr
    sub_assignment = get_assignment(assignment.trace.subtraces[first])
    get_leaf_node(sub_assignment, rest)
end

function get_leaf_node(assignment::GFTraceAssignment, addr)
    assignment.trace.primitive_calls[addr].retval
end

function get_leaf_nodes(assignment::GFTraceAssignment)
    ((key, call.retval) for (key, call) in assignment.trace.primitive_calls)
end

function get_internal_nodes(assignment::GFTraceAssignment)
    ((key, get_assignment(subtrace)) for (key, subtrace) in assignment.trace.subtraces
     if has_choices(subtrace))
end
