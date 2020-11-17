struct SwitchTrace{T1, T2, Tr} <: Trace
    kernel::Switch{T1, T2, Tr}
    p::Float64
    cond::Bool
    branch::Tr
    retval::Union{T1, T2}
    args::Tuple
    score::Float64
    noise::Float64
end

@inline get_choices(tr::SwitchTrace) = SwitchTraceChoiceMap(tr)
@inline get_retval(tr::SwitchTrace) = tr.retval
@inline get_args(tr::SwitchTrace) = tr.args
@inline get_score(tr::SwitchTrace) = tr.score
@inline get_gen_fn(tr::SwitchTrace) = tr.kernel

@inline function Base.getindex(tr::SwitchTrace, addr::Pair)
    (first, rest) = addr
    subtr = getfield(trace, first)
    subtrace[rest]
end
@inline Base.getindex(tr::SwitchTrace, addr::Symbol) = getfield(trace, addr)

function project(tr::SwitchTrace, selection::Selection)
    weight = 0.
    for k in [:cond, :branch]
        subselection = selection[k]
        weight += project(getindex(tr, k), subselection)
    end
    weight
end
project(tr::SwitchTrace, ::EmptySelection) = tr.noise

@inline function get_submap(choices::SwitchTraceChoiceMap, addr::Symbol)
    hasfield(choices, addr) || return EmptyChoiceMap()
    get_choices(getfield(choices, addr))
end
