mutable struct SwitchAssessState{T}
    weight::Float64
    retval::T
end

function process_new!(gen_fn::Switch{C, N, K, T},
                      branch_p::Float64, 
                      args::Tuple, 
                      choices::ChoiceMap, 
                      state::SwitchAssessState{T}) where {C, N, K, T}
    (weight, retval) = assess(getindex(gen_fn.mix, index), kernel_args, choices)
    state.weight += weight
    state.retval = retval
end

function assess(gen_fn::Switch{C, N, K, T}, 
                args::Tuple, 
                choices::ChoiceMap) where {C, N, K, T}
    index = args[1]
    state = SwitchAssessState{T}(0.0)
    process_new!(gen_fn, index, args[2 : end], choices, state)
    (state.weight, state.retval)
end
