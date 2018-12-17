mutable struct MapAssessState{T}
    weight::Float64
    retvals::Vector{T}
end

function process_new!(gen_fn::Map{T,U}, args::Tuple, assmt::Assignment,
                      key::Int, state::MapAssessState{T}) where {T,U}
    kernel_args = get_args_for_key(args, key)
    subassmt = get_subassmt(assmt, key)
    (weight, retval) = assess(gen_fn.kernel, kernel_args, subassmt)
    state.weight += weight
    state.retvals[key] = retval
end

function assess(gen_fn::Map{T,U}, args::Tuple, assmt::Assignment) where {T,U}
    len = length(args[1])
    state = MapAssessState{T}(0., Vector{T}(undef,len))
    for key=1:len
        process_new!(gen_fn, args, assmt, key, state)
    end
    (state.weight, PersistentVector{T}(state.retvals))
end
