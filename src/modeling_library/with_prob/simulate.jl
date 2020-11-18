mutable struct WithProbabilitySimulateState{T}
    score::Float64
    noise::Float64
    cond::Bool
    subtrace::Trace
    retval::T
    WithProbabilitySimulateState{T}(score::Float64, noise::Float64) where T = new{T}(score, noise)
end

function process!(gen_fn::WithProbability{T},
                  branch_p::Float64, 
                  args::Tuple, 
                  state::WithProbabilitySimulateState{T}) where T
    local retval::T
    flip = bernoulli(branch_p)
    state.score += logpdf(Bernoulli(), flip, branch_p)
    state.cond = flip
    subtrace = simulate(flip ? gen_fn.a : gen_fn.b, args)
    state.noise += project(subtrace, EmptySelection())
    state.subtrace = subtrace
    state.score += get_score(subtrace)
    state.retval = get_retval(subtrace)
end

function simulate(gen_fn::WithProbability{T},
                  args::Tuple) where T

    branch_p = args[1]
    state = WithProbabilitySimulateState{T}(0.0, 0.0)
    process!(gen_fn, branch_p, args[2 : end], state)
    trace = WithProbabilityTrace{T}(gen_fn, branch_p, state.cond, state.subtrace, state.retval, args[2 : end], state.score, state.noise)
    trace
end
