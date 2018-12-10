mutable struct MapInitializeState{T,U}
    score::Float64
    noise::Float64
    weight::Float64
    subtraces::Vector{U}
    retval::Vector{T}
    num_nonempty::Int
end

function process!(gen_fn::Map{T,U}, args::Tuple, assmt::Assignment,
                  key::Int, state::MapInitializeState{T,U}) where {T,U}
    local subtrace::U
    local retval::T
    kernel_args = get_args_for_key(args, key)
    subassmt = get_subassmt(assmt, key)
    (subtrace, weight) = initialize(gen_fn.kernel, kernel_args, subassmt)
    state.weight += weight 
    state.noise += project(subtrace, EmptyAddressSet())
    state.num_nonempty += (isempty(get_assignment(subtrace)) ? 0 : 1)
    state.score += get_score(subtrace)
    state.subtraces[key] = subtrace
    retval = get_retval(subtrace)
    state.retval[key] = retval
end

function initialize(gen_fn::Map{T,U}, args::Tuple, assmt::Assignment) where {T,U}
    len = length(args[1])
    state = MapInitializeState{T,U}(0., 0., 0., Vector{U}(undef,len), Vector{T}(undef,len), 0)
    # TODO check for keys that aren't valid constraints
    for key=1:len
        process!(gen_fn, args, assmt, key, state)
    end
    trace = VectorTrace{MapType,T,U}(gen_fn,
        PersistentVector{U}(state.subtraces), PersistentVector{T}(state.retval),
        args, state.score, state.noise, len, state.num_nonempty)
    (trace, state.weight)
end
