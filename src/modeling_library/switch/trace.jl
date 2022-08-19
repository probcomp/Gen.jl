# ------------ Switch trace ------------ #

struct SwitchTrace{A <: Tuple, T, U <: Trace} <: Trace
    gen_fn::GenerativeFunction{T, U}
    branch::U
    retval::T
    args::A
    score::Float64
    noise::Float64
    function SwitchTrace(gen_fn::GenerativeFunction{T, U}, 
            branch::U,
            retval::T, args::A, score::Float64,
            noise::Float64) where {T, A, U}
        new{A, T, U}(gen_fn, branch, retval, args, score, noise)
    end
end

@inline get_choices(tr::SwitchTrace) = get_choices(tr.branch)
@inline get_retval(tr::SwitchTrace) = tr.retval
@inline get_args(tr::SwitchTrace) = tr.args
@inline get_score(tr::SwitchTrace) = tr.score
@inline get_gen_fn(tr::SwitchTrace) = tr.gen_fn
@inline Base.getindex(tr::SwitchTrace, addr) = Base.getindex(tr.branch, addr)
@inline project(tr::SwitchTrace, selection::Selection) = project(tr.branch, selection)
@inline project(tr::SwitchTrace, ::EmptySelection) = tr.noise
