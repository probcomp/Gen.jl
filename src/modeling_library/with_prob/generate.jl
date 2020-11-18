mutable struct WithProbabilityGenerateState{T}
    score::Float64
    noise::Float64
    weight::Float64
    cond::Bool
    subtrace::Trace
    retval::T
    WithProbabilityGenerateState{T}(score::Float64, noise::Float64, weight::Float64) where T = new{T}(score, noise, weight)
end

function process!(gen_fn::WithProbability{T},
                  branch_p::Float64, 
                  args::Tuple, 
                  choices::ChoiceMap, 
                  state::WithProbabilityGenerateState{T}) where T

    # sample from Bernoulli with probability branch_p
    flip_d = bernoulli(branch_p)

    # check for constraints at :cond
    constrained = has_value(choices, :cond)
    !constrained && check_no_submap(choices, :cond)

    # get/constrain flip value
    constrained ? (flip = get_value(choices, :cond); state.weight += logpdf(Bernoulli(), flip, branch_p)) : flip = rand(flip_d)
    state.cond = flip

    # generate subtrace
    constraints = get_submap(choices, :branch)
    (subtrace, weight) = generate(flip ? gen_fn.a : gen_fn.b, args, constraints)
    state.subtrace = subtrace
    state.weight += weight

    # return from branch
    state.retval = get_retval(subtrace)
end

function generate(gen_fn::WithProbability{T},
                  args::Tuple, 
                  choices::ChoiceMap) where T

    branch_p = args[1]
    state = WithProbabilityGenerateState{T}(0.0, 0.0, 0.0)
    process!(gen_fn, branch_p, args[2 : end], choices, state)
    trace = WithProbabilityTrace{T}(gen_fn, branch_p, state.cond, state.subtrace, state.retval, args[2 : end], state.score, state.noise)
    (trace, state.weight)
end
