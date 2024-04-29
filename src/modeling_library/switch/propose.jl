mutable struct SwitchProposeState{T}
    choices::DynamicChoiceMap
    weight::Float64
    retval::T
    SwitchProposeState{T}(choices, weight) where T = new{T}(choices, weight)
end

function process!(rng::AbstractRNG,
                  gen_fn::Switch{C, N, K, T}, 
                  index::Int, 
                  args::Tuple, 
                  state::SwitchProposeState{T}) where {C, N, K, T}

    (submap, weight, retval) = propose(rng, getindex(gen_fn.branches, index), args)
    state.choices = submap
    state.weight += weight
    state.retval = retval
end

@inline process!(gen_fn::Switch{C, N, K, T}, index::C, args::Tuple, state::SwitchProposeState{T}) where {C, N, K, T} = process!(gen_fn, getindex(gen_fn.cases, index), args, state)

function propose(rng::AbstractRNG,
                 gen_fn::Switch{C, N, K, T}, 
                 args::Tuple) where {C, N, K, T}

    index = args[1]
    choices = choicemap()
    state = SwitchProposeState{T}(choices, 0.0)
    process!(rng, gen_fn, index, args[2:end], state)
    return state.choices, state.weight, state.retval
end
