mutable struct SwitchUpdateState{T}
    weight::Float64
    score::Float64
    noise::Float64
    prev_trace::Trace
    trace::Trace
    index::Int
    discard::DynamicChoiceMap
    updated_retdiff::Diff
    SwitchUpdateState{T}(weight::Float64, score::Float64, noise::Float64, prev_trace::Trace) where T = new{T}(weight, score, noise, subtrace)
end

function process!(gen_fn::Switch{C, N, T, K},
                  index::Int,
                  index_argdiff::Diff,
                  args::Tuple,
                  kernel_argdiffs::Tuple,
                  choices::ChoiceMap, 
                  state::SwitchUpdateState{T}) where {C, N, T, K}
    if index != getfield(state.prev_trace, :index)
        decrement = get_score(state.prev_trace)
        merged = merge!(get_choices(state.prev_trace), choices)
        new_trace, weight, retdiff, discard = update(getfield(state.prev_trace, :branch), args, kernel_argdiffs, merged)
        state.weight = weight - decrement
    else
        new_trace, weight, retdiff, discard = update(getfield(state.prev_trace, :branch), args, kernel_argdiffs, choices)
        state.weight = weight
    end
    state.trace = new_trace
    state.updated_retdiff = retdiff
    state.discard = discard
end

@inline process!(gen_fn::Switch{C, N, T, K}, index::C, index_argdiff::Diff, args::Tuple, choices::ChoiceMap, kernel_argdiffs::Tuple, state::SwitchUpdateState{T}) where {C, N, T, K} = process!(gen_fn, getindex(gen_fn.cases, index), index_argdiff, args, choices, kernel_argdiffs, state)

function update(trace::SwitchTrace{T},
                args::Tuple, 
                argdiffs::Tuple,
                choices::ChoiceMap) where T
    gen_fn = trace.gen_fn
    index, index_argdiff = args[1], argdiffs[1]
    state = SwitchUpdateState{T}(0.0, 0.0, 0.0, trace)
    process!(gen_fn, index, index_argdiff, args, kernel_argdiffs, choices, argdiffs)
    return (state.trace, state.weight, state.updated_retdiff, state.discard)
end
