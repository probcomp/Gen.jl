mutable struct MapProposeState{T}
    assmt::DynamicAssignment
    weight::Float64
    retvals::Vector{T}
end

function process_new!(gen_fn::Map{T,U}, args::Tuple, key::Int,
                      state::MapProposeState{T}) where {T,U}
    local subtrace::U
    kernel_args = get_args_for_key(args, key)
    (subassmt, weight, retval) = propose(gen_fn.kernel, kernel_args)
    set_subassmt!(state.assmt, key, subassmt)
    state.weight += weight
    state.retvals[key] = retval
end

function propose(gen_fn::Map{T,U}, args::Tuple) where {T,U}
    len = length(args[1])
    assmt = DynamicAssignment()
    state = MapProposeState{T}(assmt, 0., Vector{T}(undef,len))
    for key=1:len
        process_new!(gen_fn, args, key, state)
    end
    (state.assmt, state.weight, PersistentVector{T}(state.retvals))
end
