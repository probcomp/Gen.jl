mutable struct SwitchUpdateState{T}
    weight::Float64
    score::Float64
    noise::Float64
    prev_trace::Trace
    trace::Trace
    index::Int
    discard::DynamicChoiceMap
    updated_retdiff::Diff
    SwitchUpdateState{T}(weight::Float64, score::Float64, noise::Float64, prev_trace::Trace) where T = new{T}(weight, score, noise, prev_trace)
end

function process!(gen_fn::Switch{C, N, K, T},
                  index::Int,
                  index_argdiff::Diff,
                  args::Tuple,
                  kernel_argdiffs::Tuple,
                  choices::ChoiceMap, 
                  state::SwitchUpdateState{T}) where {C, N, K, T}
    if index != getfield(state.prev_trace, :index)
        merged = merge(get_choices(state.prev_trace), choices)
        display(merged)
        kernel_argdiffs = map(_ -> UnknownChange(), kernel_argdiffs)
        new_trace, weight, retdiff, discard = update(getfield(state.prev_trace, :branch), args, kernel_argdiffs, merged)
    else
        new_trace, weight, retdiff, discard = update(getfield(state.prev_trace, :branch), args, kernel_argdiffs, choices)
    end
    state.index = index
    state.weight = weight
    state.noise = project(new_trace, EmptySelection()) - project(state.prev_trace, EmptySelection())
    state.score = get_score(new_trace)
    state.trace = new_trace
    state.updated_retdiff = retdiff
    state.discard = discard
end

@inline process!(gen_fn::Switch{C, N, K, T}, index::C, index_argdiff::Diff, args::Tuple, kernel_argdiffs::Tuple, choices::ChoiceMap, state::SwitchUpdateState{T}) where {C, N, K, T} = process!(gen_fn, getindex(gen_fn.cases, index), index_argdiff, args, kernel_argdiffs, choices, state)

function update(trace::SwitchTrace{T},
                args::Tuple, 
                argdiffs::Tuple,
                choices::ChoiceMap) where T
    gen_fn = trace.gen_fn
    index, index_argdiff = args[1], argdiffs[1]
    state = SwitchUpdateState{T}(0.0, 0.0, 0.0, trace)
    process!(gen_fn, index, index_argdiff, args[2 : end], argdiffs[2 : end], choices, state)
    return SwitchTrace(gen_fn, state.index, state.trace, get_retval(state.trace), args, state.score, state.noise), state.weight, state.updated_retdiff, state.discard
end
