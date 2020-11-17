mutable struct SwitchSimulateState{T1, T2, Tr}
    score::Float64
    noise::Float64
    cond::Bool
    subtrace::Tr
    retval::Union{T1, T2}
    SwitchSimulateState{T1, T2, Tr}(score::Float64, noise::Float64) where {T1, T2, Tr} = new{T1, T2, Tr}(score, noise)
end

function process!(gen_fn::Switch{T1, T2, Tr}, 
                  branch_p::Float64, 
                  args::Tuple, 
                  state::SwitchSimulateState{T1, T2, Tr}) where {T1, T2, Tr}
    local subtrace::Tr
    local retval::Union{T1, T2}
    flip = bernoulli(branch_p)
    state.score += logpdf(Bernoulli(), flip, branch_p)
    state.cond = flip
    subtrace = simulate(flip ? gen_fn.a : gen_fn.b, args)
    state.noise += project(subtrace, EmptySelection())
    state.subtrace = subtrace
    state.score += get_score(subtrace)
    state.retval = get_retval(subtrace)
end

function simulate(gen_fn::Switch{T1, T2, Tr}, 
                  args::Tuple) where {T1, T2, Tr}

    branch_p = args[1]
    state = SwitchSimulateState{T1, T2, Tr}(0.0, 0.0)
    process!(gen_fn, branch_p, args[2 : end], state)
    trace = SwitchTrace{T1, T2, Tr}(gen_fn, branch_p, state.cond, state.subtrace, state.retval, args[2 : end], state.score, state.noise)
    trace
end
