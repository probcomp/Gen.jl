mutable struct SwitchGenerateState{T}
    score::Float64
    noise::Float64
    weight::Float64
    index::Int
    subtrace::Trace
    retval::T
    SwitchGenerateState{T}(score::Float64, noise::Float64, weight::Float64) where T = new{T}(score, noise, weight)
end

function process!(gen_fn::Switch{C, N, K, T},
                  index::Int, 
                  args::Tuple, 
                  choices::ChoiceMap, 
                  state::SwitchGenerateState{T}) where {C, N, K, T}

    (subtrace, weight) = generate(getindex(gen_fn.mix, index), args, choices)
    state.index = index
    state.subtrace = subtrace
    state.weight += weight
    state.retval = get_retval(subtrace)
end

@inline process!(gen_fn::Switch{C, N, K, T}, index::C, args::Tuple, choices::ChoiceMap, state::SwitchGenerateState{T}) where {C, N, K, T} = process!(gen_fn, getindex(gen_fn.cases, index), args, choices, state)

function generate(gen_fn::Switch{C, N, K, T}, 
                  args::Tuple, 
                  choices::ChoiceMap) where {C, N, K, T}

    index = args[1]
    state = SwitchGenerateState{T}(0.0, 0.0, 0.0)
    process!(gen_fn, index, args[2 : end], choices, state)
    return SwitchTrace{T}(gen_fn, state.index, state.subtrace, state.retval, args[2 : end], state.score, state.noise), state.weight
end
