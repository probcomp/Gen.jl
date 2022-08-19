mutable struct UnfoldGenerateState{T,U}
    score::Float64
    noise::Float64
    weight::Float64
    subtraces::Vector{U}
    retval::Vector{T}
    num_nonempty::Int
    state::T
end

function process!(gen_fn::Unfold{T,U}, params::Tuple, choices::ChoiceMap,
                  key::Int, state::UnfoldGenerateState{T,U}) where {T,U}
    local subtrace::U
    local new_state::T
    kernel_args = (key, state.state, params...)
    submap = get_submap(choices, key)
    (subtrace, weight) = generate(gen_fn.kernel, kernel_args, submap)
    state.weight += weight
    state.noise += project(subtrace, EmptySelection())
    state.num_nonempty += (isempty(get_choices(subtrace)) ? 0 : 1)
    state.score += get_score(subtrace)
    state.subtraces[key] = subtrace
    new_state = get_retval(subtrace)
    state.state = new_state
    state.retval[key] = new_state
end

function generate(gen_fn::Unfold{T,U}, args::Tuple, choices::ChoiceMap) where {T,U}
    len = args[1]
    init_state = args[2]
    params = args[3:end]
    state = UnfoldGenerateState{T,U}(0., 0., 0.,
        Vector{U}(undef,len), Vector{T}(undef,len), 0, init_state)
    for key=1:len
        process!(gen_fn, params, choices, key, state)
    end
    trace = VectorTrace{UnfoldType,T,U}(gen_fn,
        PersistentVector{U}(state.subtraces), PersistentVector{T}(state.retval),
        args, state.score, state.noise, len, state.num_nonempty)
    (trace, state.weight)
end
