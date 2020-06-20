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
    noise = project(subtrace, EmptySelection())
    submap = get_choices(subtrace)
    trace.isempty = trace.isempty && isempty(submap)
    trace.trie[addr] = ChoiceOrCallRecord(subtrace, score, noise, false)
    trace.score += score
    trace.noise += noise
end

###############
# GFI methods #
###############

get_args(trace::DynamicDSLTrace) = trace.args
get_retval(trace::DynamicDSLTrace) = trace.retval
get_score(trace::DynamicDSLTrace) = trace.score
get_gen_fn(trace::DynamicDSLTrace) = trace.gen_fn

## get_choices ##

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
get_submap(choices::DynamicDSLChoiceMap, addr::Pair) = _get_submap(choices, addr)

function get_submap(choices::DynamicDSLChoiceMap, addr)
    trie = choices.trie
    if has_leaf_node(trie, addr)
        # leaf node, must be a call
        call = trie[addr]
        if call.is_choice
            ValueChoiceMap(call.subtrace_or_retval)
        else
            get_choices(call.subtrace_or_retval)
        end
    elseif has_internal_node(trie, addr)
        # internal node
        subtrie = get_internal_node(trie, addr)
        DynamicDSLChoiceMap(subtrie) # see below
    else
        EmptyChoiceMap()
    end
end

function get_submaps_shallow(choices::DynamicDSLChoiceMap)
    calls_iter = (
        (key, call.is_choice ? ValueChoiceMap(call.subtrace_or_retval) : get_choices(call.subtrace_or_retval))
        for (key, call) in get_leaf_nodes(choices.trie)
    )
    internal_nodes_iter = ((key, DynamicDSLChoiceMap(trie)) for (key, trie) in get_internal_nodes(choices.trie))
    Iterators.flatten((calls_iter, internal_nodes_iter))
end

## Base.getindex ##

function _getindex(trace::DynamicDSLTrace, trie::Trie, addr::Pair)
    (first, rest) = addr
    if haskey(trie.leaf_nodes, first)
        choice_or_call = trie.leaf_nodes[first]
        if choice_or_call.is_choice
            error("Unknown address $addr; random choice at $first")
        else
            subtrace = choice_or_call.subtrace_or_retval
            return subtrace[rest]
        end
    elseif haskey(trie.internal_nodes, first)
        return _getindex(trace, trie.internal_nodes[first], rest)
    else
        error("No random choice or generative function call at address $addr")
    end
end

function _getindex(trace::DynamicDSLTrace, trie::Trie, addr)
    if haskey(trie.leaf_nodes, addr)
        choice_or_call = trie.leaf_nodes[addr]
        if choice_or_call.is_choice
            # the value of the random choice
            return choice_or_call.subtrace_or_retval
        else
            # the return value of the generative function call
            return get_retval(choice_or_call.subtrace_or_retval)
        end
    else
        error("No random choice or generative function call at address $addr")
    end
end

function Base.getindex(trace::DynamicDSLTrace, addr)
    _getindex(trace, trace.trie, addr)
end
