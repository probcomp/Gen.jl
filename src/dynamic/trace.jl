struct ChoiceRecord{T}
    retval::T
    score::Float64
end

struct CallRecord{T}
    subtrace::T
    score::Float64
    noise::Float64
end

mutable struct DynamicDSLTrace{T} <: Trace
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
    if haskey(trace.calls, addr) || haskey(trace.choices, addr)
        error("Value or subtrace already present at address $addr.
            The same address cannot be reused for multiple random choices.")
    end
    trace.choices[addr] = choice
    trace.score += choice.score
    trace.isempty = false
end

function add_call!(trace::DynamicDSLTrace, addr, subtrace)
    if haskey(trace.calls, addr) || haskey(trace.choices, addr)
        error("Value or subtrace already present at address $addr.
            The same address cannot be reused for multiple random choices.")
    end
    score = get_score(subtrace)
    noise = project(subtrace, EmptyAddressSet())
    call = CallRecord(subtrace, score, noise)
    submap = get_choices(subtrace)
    trace.isempty = trace.isempty && isempty(submap)
    trace.calls[addr] = call
    trace.score += score
    trace.noise += noise
end

# GFI methods
get_args(trace::DynamicDSLTrace) = trace.args
get_retval(trace::DynamicDSLTrace) = trace.retval
get_score(trace::DynamicDSLTrace) = trace.score
get_gen_fn(trace::DynamicDSLTrace) = trace.gen_fn

struct DynamicDSLTraceChoiceMap <: ChoiceMap
    trace::DynamicDSLTrace
    function DynamicDSLTraceChoiceMap(trace::DynamicDSLTrace)
        @assert !trace.isempty
        new(trace) 
    end
end

function get_choices(trace::DynamicDSLTrace)
    if !trace.isempty
        DynamicDSLTraceChoiceMap(trace)
    else
        EmptyChoiceMap()
    end
end

get_address_schema(::Type{DynamicDSLTraceChoiceMap}) = DynamicAddressSchema()
Base.isempty(::DynamicDSLTraceChoiceMap) = false

function _get_submap(calls::Trie, addr::Pair)
    (first, rest) = addr
    if haskey(calls, first)
        get_submap(get_choices(calls[first].subtrace), rest)
    elseif has_internal_node(calls, first)
        subcalls = get_internal_node(calls, first)
        _get_submap(subcalls, rest)
    else
        throw(KeyError(addr))
    end
end

function get_submap(choices::DynamicDSLTraceChoiceMap, addr::Pair)
    _get_submap(choices.trace.calls, addr)
end

function get_submap(choices::DynamicDSLTraceChoiceMap, addr)
    if haskey(choices.trace.calls, addr)
        call = choices.trace.calls[addr]
        get_choices(call.subtrace)
    else
        EmptyChoiceMap()
    end
end

function _has_value(calls::Trie, addr::Pair)
    (first, rest) = addr
    if haskey(calls, first)
        has_value(get_choices(calls[first].subtrace), rest)
    elseif has_internal_node(calls, first)
        subcalls = get_internal_node(calls, first)
        _has_value(subcalls, rest)
    else
        throw(KeyError(addr))
    end
end

function has_value(choices::DynamicDSLTraceChoiceMap, addr::Pair)
    if haskey(choices.trace.choices, addr)
        true
    else
        _has_value(choices.trace.calls, addr)
    end
end

function has_value(choices::DynamicDSLTraceChoiceMap, addr)
    haskey(choices.trace.choices, addr)
end

function _get_value(calls::Trie, addr::Pair)
    (first, rest) = addr
    if haskey(calls, first)
        get_value(get_choices(calls[first].subtrace), rest)
    elseif has_internal_node(calls, first)
        subcalls = get_internal_node(calls, first)
        _get_value(subcalls, rest)
    else
        throw(KeyError(addr))
    end
end

function get_value(choices::DynamicDSLTraceChoiceMap, addr::Pair)
    if haskey(choices.trace.choices, addr)
        choices.trace.choices[addr].retval
    else
        _get_value(choices.trace.calls, addr)
    end
end

function get_value(choices::DynamicDSLTraceChoiceMap, addr)
    choices.trace.choices[addr].retval
end

function get_values_shallow(choices::DynamicDSLTraceChoiceMap)
    ((key, choice.retval)
     for (key, choice) in get_leaf_nodes(choices.trace.choices))
end

function get_submaps_shallow(choices::DynamicDSLTraceChoiceMap)
    calls_iter = ((key, get_choices(call.subtrace))
        for (key, call) in get_leaf_nodes(choices.trace.calls))
    choices_iter = ((key, DynamicDSLChoicesAssmt(trie))
        for (key, trie) in get_internal_nodes(choices.trace.choices))
    Iterators.flatten((calls_iter, choices_iter))
end

# ChoiceMap wrapper that exposes sub-tries of the choices trie

struct DynamicDSLChoicesAssmt <: ChoiceMap
    choices::Trie{Any,ChoiceRecord}
end

get_address_schema(::Type{DynamicDSLChoicesAssmt}) = DynamicAddressSchema()
Base.isempty(::DynamicDSLChoicesAssmt) = false
has_value(choices::DynamicDSLChoicesAssmt, addr::Pair) = _has_value(choices, addr)
get_value(choices::DynamicDSLChoicesAssmt, addr::Pair) = _get_value(choices, addr)
get_submap(choices::DynamicDSLChoicesAssmt, addr::Pair) = _get_submap(choices, addr)

function get_submap(choices::DynamicDSLChoicesAssmt, addr)
    DynamicDSLChoicesAssmt(get_internal_node(choices.choices, addr))
end

function get_value(choices::DynamicDSLChoicesAssmt, addr)
    choices.choices[addr].retval
end

function has_value(choices::DynamicDSLChoicesAssmt, addr)
    haskey(choices.choices, addr)
end

function get_submaps_shallow(choices::DynamicDSLChoicesAssmt)
    ((key, DynamicDSLChoicesAssmt(trie))
     for (key, trie) in get_internal_nodes(choices.choices))
end

function get_values_shallow(choices::DynamicDSLChoicesAssmt)
    ((key, choice.retval)
     for (key, choice) in get_leaf_nodes(choices.choices))
end
