mutable struct SwitchGenerateState{T1, T2, Tr}
    score::Float64
    noise::Float64
    weight::Float64
    cond::Bool
    subtrace::Tr
    retval::Union{T1, T2}
    SwitchGenerateState{T1, T2, Tr}(score::Float64, noise::Float64, weight::Float64) = new{T1, T2, Tr}(score, noise, weight)
end

function process!(gen_fn::Switch{T1, T2, Tr}, 
                  branch_p::Float64, 
                  args::Tuple, 
                  choices::ChoiceMap, 
                  state::SwitchGenerateState{T1, T2, Tr}) where {T1, T2, Tr}
   
    # create flip distribution
    flip_d = Bernoulli(branch_p)

    # check for constraints at :cond
    constrained = has_value(choices, :cond)
    !constrained && check_no_submap(choices, :cond)

    # get/constrain flip value
    constrained ? (flip = get_value(choices, :cond); state.weight += logpdf(flip_d, flip)) : flip = rand(flip_d)
    state.cond = flip
  
    # generate subtrace
    constraints = get_submap(choices, :cond)
    (subtrace, weight) = generate(flip ? gen_fn.a : gen_fn.b, args, constraints)
    state.subtrace = subtrace
    state.weight += weight

    # return from branch
    get_retval(subtrace)
end

function generate(gen_fn::Switch{T1, T2, Tr}, 
                  args::Tuple, 
                  choices::ChoiceMap) where {T1, T2, Tr}

    branch_p = args[1]
    state = SwitchGenerateState{T1, T2, Tr}(0.0, 0.0, 0.0)
    process!(gen_fn, branch_p, args[2 : end], choices, state)
    trace = SwitchTrace{T1, T2, Tr}(gen_fn, branch_p, state.cond, state.subtrace, state.retval, args[2 : end], state.score, state.noise)
    (trace, state.weight)
end
