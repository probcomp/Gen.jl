mutable struct UnfoldAssessState{T}
    weight::Float64
    retvals::Vector{T}
    state::T
end

function process_new!(gen_fn::Unfold{T,U}, params::Tuple, assmt::Assignment,
                      key::Int, state::UnfoldAssessState{T}) where {T,U}
    local new_state::T
    kernel_args = (key, state.state, params...)
    subassmt = get_subassmt(assmt, key)
    (weight, new_state) = assess(gen_fn.kernel, kernel_args, subassmt)
    state.weight += weight
    state.retvals[key] = new_state
    state.state = new_state
end

function assess(gen_fn::Unfold{T,U}, args::Tuple, assmt::Assignment) where {T,U}
    len = args[1]
    init_state = args[2]
    params = args[3:end]
    state = UnfoldAssessState{T}(0., Vector{T}(undef,len), init_state)
    for key=1:len
        process_new!(gen_fn, params, assmt, key, state)
    end
    (state.weight, PersistentVector{T}(state.retvals))
end
