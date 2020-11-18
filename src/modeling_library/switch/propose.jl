mutable struct SwitchProposeState{T}
    choices::DynamicChoiceMap
    weight::Float64
    retval::T
    SwitchProposeState{T}(choices, weight) where T = new{T}(choices, weight)
end

function process_new!(gen_fn::Switch{N, K, T}, 
                      index::Int, 
                      args::Tuple, 
                      state::SwitchProposeState{T}) where {N, K, T}

    (submap, weight, retval) = propose(getindex(gen_fn.mix, index), args)
    state.choices = submap
    state.weight += weight
    state.retval = retval
end

function propose(gen_fn::Switch{N, K, T}, 
                 args::Tuple) where {N, K, T}

    index = args[1]
    choices = choicemap()
    state = SwitchProposeState{T}(choices, 0.0)
    process_new!(gen_fn, index, args[2:end], state)
    (state.choices, state.weight, state.retval)
end
