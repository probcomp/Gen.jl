mutable struct UnfoldInitializeState{T,U}
    score::Float64
    noise::Float64
    weight::Float64
    subtraces::Vector{U}
    retval::Vector{T}
    num_nonempty::Int
    state::T
end

function process!(gen_fn::Unfold{T,U}, params::Tuple, assmt::Assignment,
                  key::Int, state::UnfoldInitializeState{T,U}) where {T,U}
    local subtrace::U
    local new_state::T
    kernel_args = (key, state.state, params...)
    subassmt = get_subassmt(assmt, key)
    (subtrace, weight) = initialize(gen_fn.kernel, kernel_args, subassmt)
    state.weight += weight 
    state.noise += project(subtrace, EmptyAddressSet())
    state.num_nonempty += (isempty(get_assignment(subtrace)) ? 0 : 1)
    state.score += get_score(subtrace)
    state.subtraces[key] = subtrace
    new_state = get_retval(subtrace)
    state.state = new_state
    state.retval[key] = new_state
end

function initialize(gen_fn::Unfold{T,U}, args::Tuple, assmt::Assignment) where {T,U}
    len = args[1]
    init_state = args[2]
    params = args[3:end]
    state = UnfoldInitializeState{T,U}(0., 0., 0.,
        Vector{U}(undef,len), Vector{T}(undef,len), 0, init_state)
    for key=1:len
        process!(gen_fn, params, assmt, key, state)
    end
    trace = VectorTrace{UnfoldType,T,U}(gen_fn,
        PersistentVector{U}(state.subtraces), PersistentVector{T}(state.retval),
        args, state.score, state.noise, len, state.num_nonempty)
    (trace, state.weight)
end
