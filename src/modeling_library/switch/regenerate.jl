mutable struct SwitchRegenerateState{T}
    weight::Float64
    score::Float64
    noise::Float64
    prev_trace::Trace
    trace::Trace
    index::Int
    discard::DynamicChoiceMap
    updated_retdiff::Diff
    SwitchRegenerateState{T}(weight::Float64, score::Float64, noise::Float64, prev_trace::Trace) where T = new{T}(weight, score, noise, prev_trace)
end

function process!(gen_fn::Switch{C, N, K, T},
                  index::Int,
                  index_argdiff::Diff,
                  args::Tuple,
                  kernel_argdiffs::Tuple,
                  selection::Selection, 
                  state::SwitchRegenerateState{T}) where {C, N, K, T}
    if index != getfield(state.prev_trace, :index)
        decrement = get_score(state.prev_trace)
        kernel_argdiffs = map(_ -> UnknownChange(), kernel_argdiffs)
        new_trace, weight, retdiff, discard = regenerate(getfield(state.prev_trace, :branch), args, kernel_argdiffs, selection)
        state.weight = weight - decrement
    else
        new_trace, weight, retdiff, discard = regenerate(getfield(state.prev_trace, :branch), args, kernel_argdiffs, selection)
        state.weight = weight
    end
    state.trace = new_trace
    state.updated_retdiff = retdiff
    state.discard = discard
end

@inline process!(gen_fn::Switch{C, N, T, K}, index::C, index_argdiff::Diff, args::Tuple, kernel_argdiffs::Tuple, selection::Selection, state::SwitchRegenerateState{T}) where {C, N, T, K} = process!(gen_fn, getindex(gen_fn.cases, index), index_argdiff, args, kernel_argdiffs, selection, state)

function regenerate(trace::SwitchTrace{T},
                    args::Tuple, 
                    argdiffs::Tuple,
                    selection::Selection) where T
    gen_fn = trace.gen_fn
    index, index_argdiff = args[1], argdiffs[1]
    state = SwitchRegenerateState{T}(0.0, 0.0, 0.0, trace)
    process!(gen_fn, index, index_argdiff, args[2 : end], kernel_argdiffs[2 : end], selection, argdiffs)
    return (state.trace, state.weight, state.updated_retdiff, state.discard)
end
