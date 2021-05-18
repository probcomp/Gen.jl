mutable struct SwitchProposeState{T}
    choices::DynamicChoiceMap
    weight::Float64
    retval::T
    SwitchProposeState{T}(choices, weight) where T = new{T}(choices, weight)
end

function process!(
        gen_fn::Switch{C, N, K, T}, 
        index::Int, args::Tuple, 
        state::SwitchProposeState{T},
        parameter_context) where {C, N, K, T}
    (submap, weight, retval) = propose(
        getindex(gen_fn.branches, index), args, parameter_context)
    state.choices = submap
    state.weight += weight
    state.retval = retval
end

@inline function process!(
        gen_fn::Switch{C, N, K, T}, index::C, args::Tuple,
        state::SwitchProposeState{T},
        parameter_context) where {C, N, K, T}
    return process!(gen_fn, getindex(gen_fn.cases, index), args, state, parameter_context)
end

function propose(
        gen_fn::Switch{C, N, K, T}, args::Tuple,
        parameter_context::Dict) where {C, N, K, T}

    index = args[1]
    choices = choicemap()
    state = SwitchProposeState{T}(choices, 0.0)
    process!(gen_fn, index, args[2:end], state, parameter_context)
    return (state.choices, state.weight, state.retval)
end
