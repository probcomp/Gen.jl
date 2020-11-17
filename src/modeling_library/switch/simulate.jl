mutable struct SwitchSimulateState{T1, T2, Tr}
    score::Float64
    noise::Float64
    cond::Bool
    subtrace::Tr
    retval::Union{T1, T2}
    SwitchGenerateState{T1, T2, Tr}(score::Float64, noise::Float64) = new{T1, T2, Tr}(score, noise, weight)
end

function process!(gen_fn::Switch{T1, T2, Tr}, 
                  branch_p::Float64, 
                  args::Tuple, 
                  state::SwitchGenerateState{T1, T2, Tr}) where {T1, T2, Tr}
    local subtrace::Tr
    local retval::Union{T1, T2}
    flip_d = Bernoulli(branch_p)
    flip = rand(flip_d)
    state.score += logpdf(flip_d, flip)
    state.cond = flip
    subtrace = simulate(flip ? gen_fn.a : gen_fn.b, args)
    state.noise += project(subtrace, EmptySelection())
    state.subtrace = subtrace
    state.score += get_score(subtrace)
    get_retval(subtrace)
end

function simulate(gen_fn::Switch{T1, T2, Tr}, 
                  args::Tuple) where {T1, T2, Tr}

    branch_p = args[1]
    state = SwitchSimulateState{T1, T2, Tr}(0.0, 0.0)
    process!(gen_fn, branch_p, args[2 : end], state)
    trace = SwitchTrace{T1, T2, Tr}(gen_fn, branch_p, state.cond, state.subtrace, state.retval, args[2 : end], state.score, state.noise)
    (trace, state.weight)
end
