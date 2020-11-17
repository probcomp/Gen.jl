mutable struct SwitchProposeState{T}
    choices::DynamicChoiceMap
    weight::Float64
    retval::T
    SwitchProposeState{T}(choices, weight) where T = new{T}(choices, weight)
end

function process_new!(gen_fn::Switch{T1, T2, Tr}, 
                      branch_p::Float64, 
                      args::Tuple, 
                      state::SwitchProposeState{Union{T1, T2}}) where {T1, T2, Tr}

    flip = bernoulli(branch_p)
    (submap, weight, retval) = propose(flip ? gen_fn.a : gen_fn.b, args)
    set_value!(state.choices, :cond, flip)
    state.weight += logpdf(Bernoulli(), flip, branch_p)
    set_submap!(state.choices, :branch, submap)
    state.weight += weight
    state.retval = retval
end

function propose(gen_fn::Switch{T1, T2, Tr}, 
                 args::Tuple) where {T1, T2, Tr}

    branch_p = args[1]
    choices = choicemap()
    state = SwitchProposeState{Union{T1, T2}}(choices, 0.0)
    process_new!(gen_fn, branch_p, args[2:end], state)
    (state.choices, state.weight, state.retval)
end
