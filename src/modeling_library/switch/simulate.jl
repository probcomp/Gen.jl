mutable struct SwitchSimulateState{T}
    score::Float64
    noise::Float64
    index::Int
    subtrace::Trace
    retval::T
    SwitchSimulateState{T}(score::Float64, noise::Float64) where T = new{T}(score, noise)
end

function process!(gen_fn::Switch{N, K, T},
                  index::Int, 
                  args::Tuple, 
                  state::SwitchSimulateState{T}) where {N, K, T}
    local retval::T
    subtrace = simulate(getindex(gen_fn.mix, index), args)
    state.noise += project(subtrace, EmptySelection())
    state.subtrace = subtrace
    state.score += get_score(subtrace)
    state.retval = get_retval(subtrace)
end

function simulate(gen_fn::Switch{N, K, T},
                  args::Tuple) where {N, K, T}

    index = args[1]
    state = SwitchSimulateState{T}(0.0, 0.0)
    process!(gen_fn, index, args[2 : end], state)
    trace = SwitchTrace{T}(gen_fn, index, state.subtrace, state.retval, args[2 : end], state.score, state.noise)
    trace
end
