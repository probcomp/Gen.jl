mutable struct WithProbabilityProposeState{T}
    choices::DynamicChoiceMap
    weight::Float64
    retval::T
    WithProbabilityProposeState{T}(choices, weight) where T = new{T}(choices, weight)
end

function process_new!(gen_fn::WithProbability{T},
                      branch_p::Float64, 
                      args::Tuple, 
                      state::WithProbabilityProposeState{T}) where T

    flip = bernoulli(branch_p)
    (submap, weight, retval) = propose(flip ? gen_fn.a : gen_fn.b, args)
    set_value!(state.choices, :cond, flip)
    state.weight += logpdf(Bernoulli(), flip, branch_p)
    set_submap!(state.choices, :branch, submap)
    state.weight += weight
    state.retval = retval
end

function propose(gen_fn::WithProbability{T},
                 args::Tuple) where T

    branch_p = args[1]
    choices = choicemap()
    state = WithProbabilityProposeState{T}(choices, 0.0)
    process_new!(gen_fn, branch_p, args[2:end], state)
    (state.choices, state.weight, state.retval)
end
