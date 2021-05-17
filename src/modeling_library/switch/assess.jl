mutable struct SwitchAssessState{T}
    weight::Float64
    retval::T
    SwitchAssessState{T}(weight::Float64) where T = new{T}(weight)
end

function process!(
        gen_fn::Switch{C, N, K, T},
        index::Int, args::Tuple, 
        choices::ChoiceMap, 
        state::SwitchAssessState{T},
        parameter_context) where {C, N, K, T}
    (weight, retval) = assess(
        getindex(gen_fn.branches, index), args, choices, parameter_context)
    state.weight = weight
    state.retval = retval
end

@inline function process!(
        gen_fn::Switch{C, N, K, T}, index::C, args::Tuple,
        choices::ChoiceMap, state::SwitchAssessState{T},
        parameter_context) where {C, N, K, T}
    return process!(
        gen_fn, getindex(gen_fn.cases, index), args, choices, state, parameter_context)
end

function assess(
        gen_fn::Switch{C, N, K, T},  args::Tuple, 
        choices::ChoiceMap,
        parameter_context::Dict) where {C, N, K, T}
    index = args[1]
    state = SwitchAssessState{T}(0.0)
    process!(gen_fn, index, args[2 : end], choices, state, parameter_context)
    return (state.weight, state.retval)
end
