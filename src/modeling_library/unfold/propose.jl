mutable struct UnfoldProposeState{T}
    choices::DynamicChoiceMap # NOTE: could be read-only vector assignment
    weight::Float64
    retvals::Vector{T}
    state::T
end

function process_new!(
        gen_fn::Unfold{T,U}, params::Tuple, key::Int,
        state::UnfoldProposeState{T},
        parameter_context) where {T,U}
    local new_state::T
    kernel_args = (key, state.state, params...)
    (submap, weight, new_state) = propose(gen_fn.kernel, kernel_args)
    set_submap!(state.choices, key, submap)
    state.weight += weight
    state.retvals[key] = new_state
    state.state = new_state
end

function propose(
        gen_fn::Unfold{T,U}, args::Tuple, parameter_context::Dict) where {T,U}
    len = args[1]
    init_state = args[2]
    params = args[3:end]
    choices = choicemap()
    state = UnfoldProposeState{T}(choices, 0., Vector{T}(undef,len), init_state)
    for key=1:len
        process_new!(gen_fn, params, key, state, parameter_context)
    end
    return (state.choices, state.weight, PersistentVector{T}(state.retvals))
end
