mutable struct MapAssessState{T}
    weight::Float64
    retvals::Vector{T}
end

function process_new!(rng::AbstractRNG, gen_fn::Map{T,U}, args::Tuple, choices::ChoiceMap,
                      key::Int, state::MapAssessState{T}) where {T,U}
    kernel_args = get_args_for_key(args, key)
    submap = get_submap(choices, key)
    (weight, retval) = assess(gen_fn.kernel, kernel_args, submap)
    state.weight += weight
    state.retvals[key] = retval
end

function assess(gen_fn::Map{T,U}, args::Tuple, choices::ChoiceMap) where {T,U}
    len = length(args[1])
    state = MapAssessState{T}(0., Vector{T}(undef,len))
    for key=1:len
        # pass default rng to satisfy the interface; note, however, that it will not be used.
        process_new!(default_rng(), gen_fn, args, choices, key, state)
    end
    (state.weight, PersistentVector{T}(state.retvals))
end
