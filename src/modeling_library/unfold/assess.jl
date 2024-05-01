mutable struct UnfoldAssessState{T}
    weight::Float64
    retvals::Vector{T}
    state::T
end

function process_new!(rng::AbstractRNG, gen_fn::Unfold{T,U}, params::Tuple, choices::ChoiceMap,
                      key::Int, state::UnfoldAssessState{T}) where {T,U}
    local new_state::T
    kernel_args = (key, state.state, params...)
    submap = get_submap(choices, key)
    (weight, new_state) = assess(gen_fn.kernel, kernel_args, submap)
    state.weight += weight
    state.retvals[key] = new_state
    state.state = new_state
end

function assess(gen_fn::Unfold{T,U}, args::Tuple, choices::ChoiceMap) where {T,U}
    len = args[1]
    init_state = args[2]
    params = args[3:end]
    state = UnfoldAssessState{T}(0., Vector{T}(undef,len), init_state)
    for key=1:len
        process_new!(default_rng(), gen_fn, params, choices, key, state)
    end
    (state.weight, PersistentVector{T}(state.retvals))
end
