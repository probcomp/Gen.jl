struct CallRecord{T}
    subtrace::T
    score::Float64
    noise::Float64
end

mutable struct DynamicDSLTrace{T} <: Trace
    gen_fn::T
    trie::Trie{Any,CallRecord}
    score::Float64
    noise::Float64
    args::Tuple
    retval::Any
    function DynamicDSLTrace{T}(gen_fn::T, args) where {T}
        trie = Trie{Any,CallRecord}()
        # retval is not known yet
        new(gen_fn, trie, 0, 0, args)
    end
end

set_retval!(trace::DynamicDSLTrace, retval) = (trace.retval = retval)

has_call(trace::DynamicDSLTrace, addr) = haskey(trace.trie, addr)
get_call(trace::DynamicDSLTrace, addr) = trace.trie[addr]

function add_call!(trace::DynamicDSLTrace, addr, subtrace)
    if haskey(trace.trie, addr)
        error("Subtrace already present at address $addr.
            The same address cannot be reused for multiple random choices.")
    end
    score = get_score(subtrace)
    noise = project(subtrace, EmptySelection())
    submap = get_choices(subtrace)
    trace.trie[addr] = CallRecord(subtrace, score, noise)
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

get_choices(trace::DynamicDSLTrace) = DynamicDSLChoiceMap(trace.trie)

struct DynamicDSLChoiceMap <: AddressTree{Value}
    trie::Trie{Any,CallRecord}
end

get_address_schema(::Type{DynamicDSLChoiceMap}) = DynamicAddressSchema()
get_subtree(choices::DynamicDSLChoiceMap, addr::Pair) = _get_subtree(choices, addr)
function get_subtree(choices::DynamicDSLChoiceMap, addr)
    if haskey(choices.trie.leaf_nodes, addr)
        get_choices(choices.trie[addr].subtrace)
    elseif haskey(choices.trie.internal_nodes, addr)
        DynamicDSLChoiceMap(choices.trie.internal_nodes[addr])
    else
        EmptyChoiceMap()
    end
end

function get_subtrees_shallow(choices::DynamicDSLChoiceMap)
    leafs = ((key, get_choices(record.subtrace)) for (key, record) in get_leaf_nodes(choices.trie))
    internals = ((key, DynamicDSLChoiceMap(trie)) for (key, trie) in get_internal_nodes(choices.trie))
    Iterators.flatten((leafs, internals))
end

## Base.getindex ##

function _getindex(trace::DynamicDSLTrace, trie::Trie, addr::Pair)
    (first, rest) = addr
    if haskey(trie.leaf_nodes, first)
        return trie.leaf_nodes[first].subtrace[rest]
    elseif haskey(trie.internal_nodes, first)
        return _getindex(trace, trie.internal_nodes[first], rest)
    else
        error("No random choice or generative function call at address $addr")
    end
end

function _getindex(trace::DynamicDSLTrace, trie::Trie, addr)
    if haskey(trie.leaf_nodes, addr)
        return get_retval(trie.leaf_nodes[addr].subtrace)
    else
        error("No random choice or generative function call at address $addr")
    end
end

function Base.getindex(trace::DynamicDSLTrace, addr)
    _getindex(trace, trace.trie, addr)
end
