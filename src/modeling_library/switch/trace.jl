struct SwitchTrace{T1, T2, Tr} <: Trace
    kernel::GenerativeFunction{Union{T1, T2}, Tr}
    p::Float64
    cond::Bool
    branch::Tr
    retval::Union{T1, T2}
    args::Tuple
    score::Float64
    noise::Float64
end

@inline function get_choices(tr::SwitchTrace)
    choices = choicemap()
    set_submap!(choices, :branch, get_choices(tr.branch))
    set_value!(choices, :cond, tr.cond)
    choices
end
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
