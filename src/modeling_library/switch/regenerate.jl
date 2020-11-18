mutable struct SwitchRegenerateState{T}
    weight::Float64
    score::Float64
    noise::Float64
    prev_trace::Trace
    trace::Trace
    index::Int
    retdiff::Diff
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
        merged = get_choices(state.prev_trace)
        branch_fn = getfield(gen_fn.mix, index)
        new_trace, weight = generate(branch_fn, args, merged)
        retdiff = UnknownChange()
        weight -= get_score(state.prev_trace)
    else
        new_trace, weight, retdiff = regenerate(getfield(state.prev_trace, :branch), args, kernel_argdiffs, selection)
    end
    state.index = index
    state.weight = weight
    state.noise = project(new_trace, EmptySelection()) - project(state.prev_trace, EmptySelection())
    state.score = get_score(new_trace)
    state.trace = new_trace
    state.retdiff = retdiff
end

@inline process!(gen_fn::Switch{C, N, K, T}, index::C, index_argdiff::Diff, args::Tuple, kernel_argdiffs::Tuple, selection::Selection, state::SwitchRegenerateState{T}) where {C, N, K, T} = process!(gen_fn, getindex(gen_fn.cases, index), index_argdiff, args, kernel_argdiffs, selection, state)

function regenerate(trace::SwitchTrace{T},
                    args::Tuple, 
                    argdiffs::Tuple,
                    selection::Selection) where T
    gen_fn = trace.gen_fn
    index, index_argdiff = args[1], argdiffs[1]
    state = SwitchRegenerateState{T}(0.0, 0.0, 0.0, trace)
    process!(gen_fn, index, index_argdiff, args[2 : end], argdiffs[2 : end], selection, state)
    return SwitchTrace(gen_fn, state.index, state.trace, get_retval(state.trace), args, state.score, state.noise), state.weight, state.retdiff
end
