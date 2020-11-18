mutable struct SwitchGenerateState{T}
    score::Float64
    noise::Float64
    weight::Float64
    index::Int
    subtrace::Trace
    retval::T
    SwitchGenerateState{T}(score::Float64, noise::Float64, weight::Float64) where T = new{T}(score, noise, weight)
end

function process!(gen_fn::Switch{N, K, T},
                  index::Int, 
                  args::Tuple, 
                  choices::ChoiceMap, 
                  state::SwitchGenerateState{T}) where {N, K, T}
   
    (subtrace, weight) = generate(getindex(gen_fn.mix, index), args, choices)
    state.subtrace = subtrace
    state.weight += weight
    state.retval = get_retval(subtrace)
end

function generate(gen_fn::Switch{N, K, T}, 
                  args::Tuple, 
                  choices::ChoiceMap) where {N, K, T}

    index = args[1]
    state = SwitchGenerateState{T}(0.0, 0.0, 0.0)
    process!(gen_fn, index, args[2 : end], choices, state)
    trace = SwitchTrace{T}(gen_fn, index, state.subtrace, state.retval, args[2 : end], state.score, state.noise)
    (trace, state.weight)
end
