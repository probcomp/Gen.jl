mutable struct SwitchSimulateState{T}
    score::Float64
    noise::Float64
    index::Int
    subtrace::Trace
    retval::T
    SwitchSimulateState{T}(score::Float64, noise::Float64) where T = new{T}(score, noise)
end

function process!(rng::AbstractRNG,
                  gen_fn::Switch{C, N, K, T},
                  index::Int, 
                  args::Tuple, 
                  state::SwitchSimulateState{T}) where {C, N, K, T}
    local retval::T
    subtrace = simulate(rng, getindex(gen_fn.branches, index), args)
    state.index = index
    state.noise += project(subtrace, EmptySelection())
    state.subtrace = subtrace
    state.score += get_score(subtrace)
    state.retval = get_retval(subtrace)
end

@inline process!(gen_fn::Switch{C, N, K, T}, index::C, args::Tuple, state::SwitchSimulateState{T}) where {C, N, K, T} = process!(gen_fn, getindex(gen_fn.cases, index), args, state)

function simulate(rng::AbstractRNG,
                  gen_fn::Switch{C, N, K, T},
                  args::Tuple) where {C, N, K, T}

    index = args[1]
    state = SwitchSimulateState{T}(0.0, 0.0)
    process!(rng, gen_fn, index, args[2 : end], state)
    return SwitchTrace(gen_fn, state.subtrace, 
                       state.retval, args, 
                       state.score, state.noise)
end
