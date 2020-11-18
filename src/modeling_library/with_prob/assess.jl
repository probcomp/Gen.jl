mutable struct WithProbabilityAssessState{T}
    weight::Float64
    retval::T
end

function process_new!(gen_fn::WithProbability{T},
                      branch_p::Float64, 
                      args::Tuple, 
                      choices::ChoiceMap, 
                      state::WithProbabilityAssessState{T}) where T
    flip = get_value(choices, :cond)
    state.weight += logpdf(Bernoulli(), flip, branch_p)
    submap = get_submap(choices, :branch)
    (weight, retval) = assess(gen_fn.kernel, kernel_args, submap)
    state.weight += weight
    state.retval = retval
end

function assess(gen_fn::WithProbability{T},
                args::Tuple, 
                choices::ChoiceMap) where T
    branch_p = args[1]
    state = WithProbabilityAssessState{T}(0.0)
    process_new!(gen_fn, branch_p, args[2 : end], choices, state)
    (state.weight, state.retval)
end
