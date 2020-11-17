# Trace used by Switch combinator.
struct SwitchTrace{G1, G2, T1, T2, Tr0, Tr} <: Trace
    cond_fn::GenerativeFunction{Bool, Tr0}
    a::GenerativeFunction{T1, Tr}
    b::GenerativeFunction{T2, Tr}
    cond::Tr0
    branch::Tr
    retval::Union{T1, T2}
    args::Tuple
    score::Float64
    noise::Float64
end

function SwitchTrace{G1, G2, T1, T2, Tr0, Tr1, Tr2}(cond::Generativefunction{Bool},
                                                    a::GenerativeFunction{T1},
                                                    b::GenerativeFunction{T2},
                                                    cond_subtrace::Tr0,
                                                    branch_subtrace::Union{Tr1, Tr2},
                                                    retval::Union{T1, T2},
                                                    args::Tuple,
                                                    score::Float64
                                                    noise::Float64) where {G1, G2, T1, T2, Tr0, Tr1, Tr2}
end

@inline get_choices(tr::SwitchTrace) = SwitchTraceChoiceMap(tr)
@inline get_retval(tr::SwitchTrace) = tr.retval
@inline get_args(tr::SwitchTrace) = tr.args
@inline get_score(tr::SwitchTrace) = tr.score
# TODO. @inline get_gen_fn(tr::SwitchTrace) = tr.gen_fn

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
