######################
# static assignment #
######################

struct StaticChoiceMap{Addrs, SubmapTypes} <: ChoiceMap
    submaps::NamedTuple{Addrs, SubmapTypes}
    function StaticChoiceMap(submaps::NamedTuple{Addrs, SubmapTypes}) where {Addrs, SubmapTypes <: NTuple{n, ChoiceMap} where n}
        new{Addrs, SubmapTypes}(submaps)
    end
end

function StaticChoiceMap(;addrs_to_vals_and_maps...)
    addrs = Tuple(addr for (addr, val_or_map) in addrs_to_vals_and_maps)
    maps = Tuple(val_or_map isa ChoiceMap ? val_or_map : ValueChoiceMap(val_or_map) for (addr, val_or_map) in addrs_to_vals_and_maps)
    StaticChoiceMap(NamedTuple{addrs}(maps))
end

@inline get_submaps_shallow(choices::StaticChoiceMap) = pairs(choices.submaps)
@inline get_submap(choices::StaticChoiceMap, addr::Pair) = _get_submap(choices, addr)
@inline get_submap(choices::StaticChoiceMap, addr::Symbol) = static_get_submap(choices, Val(addr))

@generated function static_get_submap(choices::StaticChoiceMap{Addrs, SubmapTypes}, ::Val{A}) where {A, Addrs, SubmapTypes}
    if A in Addrs
        quote choices.submaps[A] end
    else
        quote EmptyChoiceMap() end
    end
end
@inline static_get_submap(::EmptyChoiceMap, ::Val) = EmptyChoiceMap()

@inline static_get_value(choices::StaticChoiceMap, v::Val) = get_value(static_get_submap(choices, v))
@inline static_get_value(::EmptyChoiceMap, ::Val) = throw(ChoiceMapGetValueError())

# convert a nonvalue choicemap all of whose top-level-addresses 
# are symbols into a staticchoicemap at the top level
function StaticChoiceMap(other::ChoiceMap)
    keys_and_nodes = collect(get_submaps_shallow(other))
    if length(keys_and_nodes) > 0
        (addrs::NTuple{n, Symbol} where {n}, submaps) = zip(keys_and_nodes...)
    else
        addrs = ()
        submaps = ()
    end
    StaticChoiceMap(NamedTuple{addrs}(submaps))
end
StaticChoiceMap(other::ValueChoiceMap) = error("Cannot convert a ValueChoiceMap to a StaticChoiceMap")
StaticChoiceMap(::NamedTuple{(),Tuple{}}) = EmptyChoiceMap()

# TODO: deep conversion to static choicemap

"""
    choices = pair(choices1::ChoiceMap, choices2::ChoiceMap, key1::Symbol, key2::Symbol)

Return an assignment that contains `choices1` as a sub-assignment under `key1`
and `choices2` as a sub-assignment under `key2`.
"""
function pair(choices1::ChoiceMap, choices2::ChoiceMap, key1::Symbol, key2::Symbol)
    StaticChoiceMap(NamedTuple{(key1, key2)}((choices1, choices2)))
end

"""
    (choices1, choices2) = unpair(choices::ChoiceMap, key1::Symbol, key2::Symbol)

Return the two sub-assignments at `key1` and `key2`, one or both of which may be empty.

It is an error if there are any submaps at keys other than `key1` and `key2`.
"""
function unpair(choices::ChoiceMap, key1::Symbol, key2::Symbol)
    if length(collect(get_submaps_shallow(choices))) != 2
        error("Not a pair")
    end
    (get_submap(choices, key1), get_submap(choices, key2))
end

@generated function Base.merge(choices1::StaticChoiceMap{Addrs1, SubmapTypes1},
    choices2::StaticChoiceMap{Addrs2, SubmapTypes2}) where {Addrs1, Addrs2, SubmapTypes1, SubmapTypes2}
    
    addr_to_type1 = Dict{Symbol, Type{<:ChoiceMap}}()
    addr_to_type2 = Dict{Symbol, Type{<:ChoiceMap}}()
    for (i, addr) in enumerate(Addrs1)
        addr_to_type1[addr] = SubmapTypes1.parameters[i]
    end
    for (i, addr) in enumerate(Addrs2)
        addr_to_type2[addr] = SubmapTypes2.parameters[i]
    end

    merged_addrs = Tuple(union(Set(Addrs1), Set(Addrs2)))
    submap_exprs = []

    for addr in merged_addrs
        type1 = get(addr_to_type1, addr, EmptyChoiceMap)
        type2 = get(addr_to_type2, addr, EmptyChoiceMap)
        if ((type1 <: ValueChoiceMap && type2 != EmptyChoiceMap)
            || (type2 <: ValueChoiceMap && type1 != EmptyChoiceMap))
           error( "One choicemap has a value at address $addr; the other is nonempty at $addr.  Cannot merge.")
        end
        if type1 <: EmptyChoiceMap
            push!(submap_exprs, 
                quote choices2.submaps.$addr end
            )
        elseif type2 <: EmptyChoiceMap
            push!(submap_exprs,
                quote choices1.submaps.$addr end
            )
        else
            push!(submap_exprs,
                quote merge(choices1.submaps.$addr, choices2.submaps.$addr) end
            )
        end
    end

    quote
        StaticChoiceMap(NamedTuple{$merged_addrs}(($(submap_exprs...),)))
    end
end

@generated function _from_array(proto_choices::StaticChoiceMap{Addrs, SubmapTypes},
    arr::Vector{T}, start_idx::Int) where {T, Addrs, SubmapTypes}

    perm = sortperm(collect(Addrs))
    sorted_addrs = Addrs[perm]
    submap_var_names = Vector{Symbol}(undef, length(sorted_addrs))

    exprs = [quote idx = start_idx end]

    for (idx, addr) in zip(perm, sorted_addrs)
        submap_var_name = gensym(addr)
        submap_var_names[idx] = submap_var_name
        push!(exprs,
            quote
                (n_read, $submap_var_name) = _from_array(proto_choices.submaps.$addr, arr, idx)
                idx += n_read
            end
        )
    end

    quote
        $(exprs...)
        submaps = NamedTuple{Addrs}(( $(submap_var_names...), ))
        choices = StaticChoiceMap(submaps)
        (idx - start_idx, choices)
    end
end

function get_address_schema(::Type{StaticChoiceMap{Addrs, SubmapTypes}}) where {Addrs, SubmapTypes}
    StaticAddressSchema(Set(Addrs))
end

export StaticChoiceMap
export pair, unpair
export static_get_submap, static_get_value