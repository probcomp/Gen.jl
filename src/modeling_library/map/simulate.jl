mutable struct MapSimulateState{T,U}
    score::Float64
    noise::Float64
    subtraces::Vector{U}
    retval::Vector{T}
    num_nonempty::Int
end

function process!(rng::AbstractRNG, gen_fn::Map{T,U}, args::Tuple,
                  key::Int, state::MapSimulateState{T,U}) where {T,U}
    local subtrace::U
    local retval::T
    kernel_args = get_args_for_key(args, key)
    subtrace = simulate(rng, gen_fn.kernel, kernel_args)
    state.noise += project(subtrace, EmptySelection())
    state.num_nonempty += (isempty(get_choices(subtrace)) ? 0 : 1)
    state.score += get_score(subtrace)
    state.subtraces[key] = subtrace
    retval = get_retval(subtrace)
    state.retval[key] = retval
end

function simulate(rng::AbstractRNG, gen_fn::Map{T,U}, args::Tuple) where {T,U}
    len = length(args[1])
    state = MapSimulateState{T,U}(0., 0., Vector{U}(undef,len), Vector{T}(undef,len), 0)
    for key=1:len
        process!(rng, gen_fn, args, key, state)
    end
    VectorTrace{MapType,T,U}(gen_fn,
        PersistentVector{U}(state.subtraces), PersistentVector{T}(state.retval),
        args, state.score, state.noise, len, state.num_nonempty)
end
