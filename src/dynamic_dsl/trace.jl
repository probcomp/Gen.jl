struct ChoiceRecord{T}
    retval::T
    score::Float64
end

struct CallRecord{T}
    subtrace::T
    score::Float64
    noise::Float64
end

mutable struct DynamicDSLTrace{T}
    gen_fn::T
    choices::Trie{Any,ChoiceRecord}
    calls::Trie{Any,CallRecord}
    isempty::Bool
    score::Float64
    noise::Float64
    args::Tuple
    retval::Any
    function DynamicDSLTrace{T}(gen_fn::T, args) where {T}
        choices = Trie{Any,ChoiceRecord}()
        calls = Trie{Any,CallRecord}()
        # retval is not known yet
        new(gen_fn, choices, calls, true, 0, 0, args)
    end
end

DynamicDSLTrace(gen_fn::T, args) where {T} = DynamicDSLTrace{T}(gen_fn, args)

set_retval!(trace::DynamicDSLTrace, retval) = (trace.retval = retval)
has_choice(trace::DynamicDSLTrace, addr) = haskey(trace.choices, addr)
has_call(trace::DynamicDSLTrace, addr) = haskey(trace.calls, addr)
get_choice(trace::DynamicDSLTrace, addr) = trace.choices[addr]
get_call(trace::DynamicDSLTrace, addr) = trace.calls[addr]

function add_choice!(trace::DynamicDSLTrace, addr, choice::ChoiceRecord)
    @assert !haskey(trace.calls, addr)
    @assert !haskey(trace.choices, addr)
    trace.choices[addr] = choice
    trace.score += choice.score
    trace.isempty = false
end

function add_call!(trace::DynamicDSLTrace, addr, subtrace)
    @assert !haskey(trace.calls, addr)
    @assert !haskey(trace.choices, addr)
    score = get_score(subtrace)
    noise = project(subtrace, EmptyAddressSet())
    call = CallRecord(subtrace, score, noise)
    subassmt = get_assignment(subtrace)
    trace.isempty = trace.isempty && isempty(subassmt)
    trace.calls[addr] = call
    trace.score += score
    trace.noise += noise
end

# GFI methods
get_args(trace::DynamicDSLTrace) = trace.args
get_retval(trace::DynamicDSLTrace) = trace.retval
get_score(trace::DynamicDSLTrace) = trace.score

struct DynamicDSLTraceAssignment <: Assignment
    trace::DynamicDSLTrace
    function DynamicDSLTraceAssignment(trace::DynamicDSLTrace)
        @assert !trace.isempty
        new(trace) 
    end
end

function get_assignment(trace::DynamicDSLTrace)
    if !trace.isempty
        DynamicDSLTraceAssignment(trace)
    else
        EmptyAssignment()
    end
end

get_address_schema(::Type{DynamicDSLTraceAssignment}) = DynamicAddressSchema()
Base.isempty(::DynamicDSLTraceAssignment) = false

function _get_subassmt(calls::Trie, addr::Pair)
    (first, rest) = addr
    if haskey(calls, first)
        get_subassmt(get_assignment(calls[first].subtrace), rest)
    elseif has_internal_node(calls, first)
        subcalls = get_internal_node(calls, first)
        _get_subassmt(subcalls, rest)
    else
        throw(KeyError(addr))
    end
end

function get_subassmt(assmt::DynamicDSLTraceAssignment, addr::Pair)
    _get_subassmt(assmt.trace.calls, addr)
end

function get_subassmt(assmt::DynamicDSLTraceAssignment, addr)
    if haskey(assmt.trace.calls, addr)
        call = assmt.trace.calls[addr]
        get_assignment(call.subtrace)
    else
        EmptyAssignment()
    end
end

function _has_value(calls::Trie, addr::Pair)
    (first, rest) = addr
    if haskey(calls, first)
        has_value(get_assignment(calls[first].subtrace), rest)
    elseif has_internal_node(calls, first)
        subcalls = get_internal_node(calls, first)
        _has_value(subcalls, rest)
    else
        throw(KeyError(addr))
    end
end

function has_value(assmt::DynamicDSLTraceAssignment, addr::Pair)
    if haskey(assmt.trace.choices, addr)
        true
    else
        _has_value(assmt.trace.calls, addr)
    end
end

function has_value(assmt::DynamicDSLTraceAssignment, addr)
    haskey(assmt.trace.choices, addr)
end

function _get_value(calls::Trie, addr::Pair)
    (first, rest) = addr
    if haskey(calls, first)
        get_value(get_assignment(calls[first].subtrace), rest)
    elseif has_internal_node(calls, first)
        subcalls = get_internal_node(calls, first)
        _get_value(subcalls, rest)
    else
        throw(KeyError(addr))
    end
end

function get_value(assmt::DynamicDSLTraceAssignment, addr::Pair)
    if haskey(assmt.trace.choices, addr)
        assmt.trace.choices[addr].retval
    else
        _get_value(assmt.trace.calls, addr)
    end
end

function get_value(assmt::DynamicDSLTraceAssignment, addr)
    assmt.trace.choices[addr].retval
end

function get_values_shallow(assmt::DynamicDSLTraceAssignment)
    ((key, choice.retval)
     for (key, choice) in get_leaf_nodes(assmt.trace.choices))
end

function get_subassmts_shallow(assmt::DynamicDSLTraceAssignment)
    calls_iter = ((key, get_assignment(call.subtrace))
        for (key, call) in get_leaf_nodes(assmt.trace.calls))
    choices_iter = ((key, DynamicDSLChoicesAssmt(subchoices))
        for (key, trie) in get_internal_nodes(assmt.trace.choices))
    Iterators.flatten((calls_iter, choices_iter))
end

# Assignment wrapper that exposes sub-tries of the choices trie

struct DynamicDSLChoicesAssmt <: Assignment
    choices::Trie{Any,ChoiceRecord}
end

get_address_schema(::Type{DynamicDSLChoicesAssmt}) = DynamicAddressSchema()
Base.isempty(::DynamicDSLChoicesAssmt) = false
has_value(assmt::DynamicDSLChoicesAssmt, addr::Pair) = _has_value(assmt, addr)
get_value(assmt::DynamicDSLChoicesAssmt, addr::Pair) = _get_value(assmt, addr)
get_subassmt(assmt::DynamicDSLChoicesAssmt, addr::Pair) = _get_subassmt(assmt, addr)

function get_subassmt(assmt::DynamicDSLChoicesAssmt, addr)
    DynamicDSLChoicesAssmt(get_internal_node(assmt.trace.choices, addr))
end

function get_value(assmt::DynamicDSLChoicesAssmt, addr)
    assmt.trace.choices[addr].retval
end

function has_value(assmt::DynamicDSLChoicesAssmt, addr)
    haskey(assmt.trace.choices, addr)
end

function get_subassmts_shallow(assmt::DynamicDSLChoicesAssmt)
    ((key, DynamicDSLChoicesAssmt(trie)
     for (key, trie) in get_internal_nodes(assmt.trace.choices)))
end

function get_values_shallow(assmt::DynamicDSLChoicesAssmt)
    ((key, choice.retval)
     for choice in get_leaf_nodes(assmt.trace.choices))
end
