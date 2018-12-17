mutable struct UnfoldProposeState{T}
    assmt::DynamicAssignment # NOTE: could be read-only vector assignment
    weight::Float64
    retvals::Vector{T}
    state::T
end

function process_new!(gen_fn::Unfold{T,U}, params::Tuple, key::Int,
                      state::UnfoldProposeState{T}) where {T,U}
    local new_state::T
    kernel_args = (key, state.state, params...)
    (subassmt, weight, new_state) = propose(gen_fn.kernel, kernel_args)
    set_subassmt!(state.assmt, key, subassmt)
    state.weight += weight
    state.retvals[key] = new_state
    state.state = new_state
end

function propose(gen_fn::Unfold{T,U}, args::Tuple) where {T,U}
    len = args[1]
    init_state = args[2]
    params = args[3:end]
    assmt = DynamicAssignment()
    state = UnfoldProposeState{T}(assmt, 0., Vector{T}(undef,len), init_state)
    for key=1:len
        process_new!(gen_fn, params, key, state)
    end
    (state.assmt, state.weight, PersistentVector{T}(state.retvals))
end
