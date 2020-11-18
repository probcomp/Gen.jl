mutable struct WithProbabilityUpdateState{T}
    weight::Float64
    score::Float64
    noise::Float64
    cond::Bool
    prev_trace::Trace
    trace::Trace
    discard::DynamicChoiceMap
    updated_retdiff::Diff
end

function process!(gen_fn::WithProbability{T}
                  branch_p::Float64,
                  args::Tuple,
                  kernel_argdiffs::Tuple,
                  choices::ChoiceMap, 
                  state::WithProbabilityUpdateState{T}) where T
end

function update(trace::WithProbability{T},
                args::Tuple, 
                argdiffs::Tuple,
                choices::ChoiceMap) where T
    gen_fn = trace.gen_fn
    branch_p, branch_p_diff = args[1], argdiffs[1]
    return (new_trace, state.weight, retdiff, discard)
end
