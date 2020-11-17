mutable struct SwitchAssessState{T}
    weight::Float64
    retval::T
end

function process_new!(gen_fn::Switch{T1, T2, Tr}, 
                      branch_p::Float64, 
                      args::Tuple, 
                      choices::ChoiceMap, 
                      state::SwitchAssessState{Union{T1, T2}}) where {T1, T2, Tr}
    flip = get_value(choices, :cond)
    state.weight += logpdf(Bernoulli(), flip, branch_p)
    submap = get_submap(choices, :branch)
    (weight, retval) = assess(gen_fn.kernel, kernel_args, submap)
    state.weight += weight
    state.retval = retval
end

function assess(gen_fn::Switch{T1, T2, Tr}, 
                args::Tuple, 
                choices::ChoiceMap) where {T1, T2, Tr}
    branch_p = args[1]
    state = SwitchAssessState{Union{T1, T2}}(0.0)
    process_new!(gen_fn, branch_p, args[2 : end], choices, state)
    (state.weight, state.retval)
end
