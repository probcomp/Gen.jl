mutable struct MapProposeState{T}
    choices::DynamicChoiceMap
    weight::Float64
    retvals::Vector{T}
end

function process_new!(
        gen_fn::Map{T,U}, args::Tuple, key::Int,
        state::MapProposeState{T},
        parameter_context) where {T,U}
    local subtrace::U
    kernel_args = get_args_for_key(args, key)
    (submap, weight, retval) = propose(gen_fn.kernel, kernel_args, parameter_context)
    set_submap!(state.choices, key, submap)
    state.weight += weight
    state.retvals[key] = retval
end

function propose(
        gen_fn::Map{T,U}, args::Tuple, parameter_context::Dict) where {T,U}
    len = length(args[1])
    choices = choicemap()
    state = MapProposeState{T}(choices, 0., Vector{T}(undef,len))
    for key=1:len
        process_new!(gen_fn, args, key, state, parameter_context)
    end
    return (state.choices, state.weight, PersistentVector{T}(state.retvals))
end
