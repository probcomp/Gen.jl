mutable struct SwitchGenerateState{T}
    score::Float64
    noise::Float64
    weight::Float64
    index::Int
    subtrace::Trace
    retval::T
    SwitchGenerateState{T}(score::Float64, noise::Float64, weight::Float64) where T = new{T}(score, noise, weight)
end

function process!(rng::AbstractRNG,
                  gen_fn::Switch{C, N, K, T},
                  index::Int, 
                  args::Tuple, 
                  choices::ChoiceMap, 
                  state::SwitchGenerateState{T}) where {C, N, K, T}

    (subtrace, weight) = generate(rng, getindex(gen_fn.branches, index), args, choices)
    state.index = index
    state.subtrace = subtrace
    state.weight += weight
    state.retval = get_retval(subtrace)
end

@inline process!(gen_fn::Switch{C, N, K, T}, index::C, args::Tuple, choices::ChoiceMap, state::SwitchGenerateState{T}) where {C, N, K, T} = process!(gen_fn, getindex(gen_fn.cases, index), args, choices, state)

function generate(rng::AbstractRNG,
                  gen_fn::Switch{C, N, K, T}, 
                  args::Tuple, 
                  choices::ChoiceMap) where {C, N, K, T}

    index = args[1]
    state = SwitchGenerateState{T}(0.0, 0.0, 0.0)
    process!(rng, gen_fn, index, args[2 : end], choices, state)
    return SwitchTrace(gen_fn, state.subtrace, 
                       state.retval, args, 
                       state.score, state.noise), state.weight
end
