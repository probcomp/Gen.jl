# ------------ WithProbability trace ------------ #

struct WithProbabilityTrace{T1, T2, Tr} <: Trace
    gen_fn::GenerativeFunction{Union{T1, T2}, Tr}
    p::Float64
    cond::Bool
    branch::Tr
    retval::Union{T1, T2}
    args::Tuple
    score::Float64
    noise::Float64
end

@inline function get_choices(tr::WithProbabilityTrace)
    choices = choicemap()
    set_submap!(choices, :branch, get_choices(tr.branch))
    set_value!(choices, :cond, tr.cond)
    choices
end
@inline get_retval(tr::WithProbabilityTrace) = tr.retval
@inline get_args(tr::WithProbabilityTrace) = tr.args
@inline get_score(tr::WithProbabilityTrace) = tr.score
@inline get_gen_fn(tr::WithProbabilityTrace) = tr.gen_fn

@inline function Base.getindex(tr::WithProbabilityTrace, addr::Pair)
    (first, rest) = addr
    subtr = getfield(trace, first)
    subtrace[rest]
end
@inline Base.getindex(tr::WithProbabilityTrace, addr::Symbol) = getfield(trace, addr)

function project(tr::WithProbabilityTrace, selection::Selection)
    sum(map([:cond, :branch]) do k
            subselection = selection[k]
            project(getindex(tr, k), subselection)
        end)
end
project(tr::WithProbabilityTrace, ::EmptySelection) = tr.noise

# ------------ Switch trace ------------ #

struct SwitchTrace{T} <: Trace
    gen_fn::GenerativeFunction{T}
    index::Int
    branch::Trace
    retval::T
    args::Tuple
    score::Float64
    noise::Float64
end

@inline get_choices(tr::SwitchTrace) = get_choices(tr.branch)
@inline get_retval(tr::SwitchTrace) = tr.retval
@inline get_args(tr::SwitchTrace) = tr.args
@inline get_score(tr::SwitchTrace) = tr.score
@inline get_gen_fn(tr::SwitchTrace) = tr.gen_fn

@inline function Base.getindex(tr::SwitchTrace, addr::Pair)
    (first, rest) = addr
    subtr = getfield(trace, first)
    subtrace[rest]
end
@inline Base.getindex(tr::SwitchTrace, addr::Symbol) = getfield(trace, addr)

@inline project(tr::SwitchTrace, selection::Selection) = project(tr.branch, selection)
@inline project(tr::SwitchTrace, ::EmptySelection) = tr.noise
