mutable struct MapAssessState{T}
    weight::Float64
    retvals::Vector{T}
end

function process_new!(
        gen_fn::Map{T,U}, args::Tuple, choices::ChoiceMap,
        key::Int, state::MapAssessState{T},
        parameter_context) where {T,U}
    kernel_args = get_args_for_key(args, key)
    submap = get_submap(choices, key)
    (weight, retval) = assess(gen_fn.kernel, kernel_args, submap, parameter_context)
    state.weight += weight
    state.retvals[key] = retval
end

function assess(
        gen_fn::Map{T,U}, args::Tuple, choices::ChoiceMap,
        parameter_context::Dict) where {T,U}
    len = length(args[1])
    state = MapAssessState{T}(0., Vector{T}(undef,len))
    for key=1:len
        process_new!(gen_fn, args, choices, key, state, parameter_context)
    end
    return (state.weight, PersistentVector{T}(state.retvals))
end
