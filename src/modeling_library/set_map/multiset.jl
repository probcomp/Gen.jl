import FunctionalCollections

export MultiSet, remove_one, setmap

struct MultiSet{T}
    counts::PersistentHashMap{T, Int}
    len::Int
end
function MultiSet{T}() where {T}
    MultiSet{T}(PersistentHashMap{T, Int}(), 0)
end
MultiSet() = MultiSet{Any}()

function MultiSet(vals::Vector{T}) where T
    ms = MultiSet{T}()
    for val in vals
        ms = push(ms, val)
    end
    ms
end

Base.length(ms::MultiSet) = ms.len
function Base.:(==)(ms1::MultiSet, ms2::MultiSet)
    if length(ms1) !== length(ms2); return false; end;
    for (key, cnt) in ms1.counts
        if !haskey(ms2.counts, key); return false; end;
        if ms2.counts[key] !== cnt; return false; end;
    end
    for (key, _) in ms2.counts
        if !haskey(ms1.counts, key); return false; end;
    end
    return true
end
function FunctionalCollections.push(ms::MultiSet{T}, el::T) where T
    if haskey(ms.counts, el)
        return MultiSet{T}(assoc(ms.counts, el, ms.counts[el] + 1), ms.len + 1)
    else
        return MultiSet{T}(assoc(ms.counts, el, 1), ms.len + 1)
    end
end
function remove_one(ms::MultiSet{T}, el::T) where T
    cnt = ms.counts[el]
    if cnt == 1
        return MultiSet{T}(dissoc(ms.counts, el), ms.len - 1)
    else
        return MultiSet{T}(assoc(ms.counts, el, cnt - 1), ms.len - 1)
    end
end
function FunctionalCollections.disj(ms::MultiSet{T}, el::T) where T
    cnt = ms.counts[el]
    return MultiSet{T}(dissoc(ms.counts, el), ms.len - cnt)
end
Base.in(el::T, ms::MultiSet{T}) where T = haskey(ms.counts, el)
function Base.iterate(ms::MultiSet{T}) where T
    i = iterate(ms.counts)
    if i === nothing; return nothing; end;
    ((key, cnt), st) = i
    return Base.iterate(ms, (key, cnt, st))
end
function Base.iterate(ms::MultiSet{T}, (key, cnt, st)) where T
    if cnt == 0
        i = iterate(ms.counts, st)
        if i === nothing; return nothing; end;
        ((key, cnt), st) = i
        return Base.iterate(ms, (key, cnt, st))
    else
        return (key, (key, cnt-1, st))
    end
end

function setmap(f, set)
    vals = [f(el) for el in set]
    MultiSet(vals)
end