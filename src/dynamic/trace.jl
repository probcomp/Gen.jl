struct ChoiceRecord{T}
    retval::T
    score::Float64
end

struct CallRecord{T}
    subtrace::T
    score::Float64
    noise::Float64
end

struct ChoiceOrCallRecord{T}
    subtrace_or_retval::T
    score::Float64
    noise::Float64 # if choice then NaN
    is_choice::Bool
end

function ChoiceRecord(record::ChoiceOrCallRecord)
    if !record.is_choice
        error("Found call but expected choice")
    end
    ChoiceRecord(record.subtrace_or_retval, record.score)
end

function CallRecord(record::ChoiceOrCallRecord)
    if record.is_choice
        error("Found choice but expected call")
    end
    CallRecord(record.subtrace_or_retval, record.score, record.noise)
end
    
mutable struct DynamicDSLTrace{T} <: Trace
    gen_fn::T
    trie::Trie{Any,ChoiceOrCallRecord}
    isempty::Bool
    score::Float64
    noise::Float64
    args::Tuple
    retval::Any
    function DynamicDSLTrace{T}(gen_fn::T, args) where {T}
        trie = Trie{Any,ChoiceOrCallRecord}()
        # retval is not known yet
        new(gen_fn, trie, true, 0, 0, args)
    end
end

DynamicDSLTrace(gen_fn::T, args) where {T} = DynamicDSLTrace{T}(gen_fn, args)

set_retval!(trace::DynamicDSLTrace, retval) = (trace.retval = retval)

function has_choice(trace::DynamicDSLTrace, addr)
    haskey(trace.trie, addr) && trace.trie[addr].is_choice
end

function has_call(trace::DynamicDSLTrace, addr)
    haskey(trace.trie, addr) && !trace.trie[addr].is_choice
end

function get_choice(trace::DynamicDSLTrace, addr)
    choice = trace.trie[addr]
    if !choice.is_choice
        throw(KeyError(addr))
    end
    ChoiceRecord(choice)
end

function get_call(trace::DynamicDSLTrace, addr)
    call = trace.trie[addr]
    if call.is_choice
        throw(KeyError(addr))
    end
    CallRecord(call)
end

function add_choice!(trace::DynamicDSLTrace, addr, retval, score)
    if haskey(trace.trie, addr)
        error("Value or subtrace already present at address $addr.
            The same address cannot be reused for multiple random choices.")
    end
    trace.trie[addr] = ChoiceOrCallRecord(retval, score, NaN, true)
    trace.score += score
    trace.isempty = false
end

function add_call!(trace::DynamicDSLTrace, addr, subtrace)
    if haskey(trace.trie, addr)
        error("Value or subtrace already present at address $addr.
            The same address cannot be reused for multiple random choices.")
    end
    score = get_score(subtrace)
    noise = project(subtrace, EmptyAddressSet())
    submap = get_choices(subtrace)
    trace.isempty = trace.isempty && isempty(submap)
    trace.trie[addr] = ChoiceOrCallRecord(subtrace, score, noise, false)
    trace.score += score
    trace.noise += noise
end

# GFI methods
get_args(trace::DynamicDSLTrace) = trace.args
get_retval(trace::DynamicDSLTrace) = trace.retval
get_score(trace::DynamicDSLTrace) = trace.score
get_gen_fn(trace::DynamicDSLTrace) = trace.gen_fn

function get_choices(trace::DynamicDSLTrace)
    if !trace.isempty
        DynamicDSLChoiceMap(trace.trie) # see below
    else
        EmptyChoiceMap()
    end
end

struct DynamicDSLChoiceMap <: ChoiceMap
    trie::Trie{Any,ChoiceOrCallRecord}
end

get_address_schema(::Type{DynamicDSLChoiceMap}) = DynamicAddressSchema()
Base.isempty(::DynamicDSLChoiceMap) = false # TODO not necessarily true
has_value(choices::DynamicDSLChoiceMap, addr::Pair) = _has_value(choices, addr)
get_value(choices::DynamicDSLChoiceMap, addr::Pair) = _get_value(choices, addr)
get_submap(choices::DynamicDSLChoiceMap, addr::Pair) = _get_submap(choices, addr)

function get_submap(choices::DynamicDSLChoiceMap, addr)
    trie = choices.trie
    if has_leaf_node(trie, addr)
        # leaf node, must be a call
        call = trie[addr]
        if call.is_choice
            throw(KeyError(addr))
        end
        get_choices(call.subtrace_or_retval)
    elseif has_internal_node(trie, addr)
        # internal node
        subtrie = get_internal_node(trie, addr)
        DynamicDSLChoiceMap(subtrie) # see below
    else
        EmptyChoiceMap()
    end
end

function has_value(choices::DynamicDSLChoiceMap, addr)
    trie = choices.trie
    has_leaf_node(trie, addr) && trie[addr].is_choice
end

function get_value(choices::DynamicDSLChoiceMap, addr)
    trie = choices.trie
    choice = trie[addr]
    if !choice.is_choice
        throw(KeyError(addr))
    end
    choice.subtrace_or_retval
end

function get_values_shallow(choices::DynamicDSLChoiceMap)
    ((key, choice.subtrace_or_retval)
     for (key, choice) in get_leaf_nodes(choices.trie)
     if choice.is_choice)
end

function get_submaps_shallow(choices::DynamicDSLChoiceMap)
    calls_iter = ((key, get_choices(call.subtrace_or_retval))
        for (key, call) in get_leaf_nodes(choices.trie)
        if !call.is_choice)
    internal_nodes_iter = ((key, DynamicDSLChoiceMap(trie))
        for (key, trie) in get_internal_nodes(choices.trie))
    Iterators.flatten((calls_iter, internal_nodes_iter))
end
