mutable struct MapAssessState{T,U}
    score::Float64
    subtraces::Vector{U}
    retvals::Vector{T}
    num_has_choices::Int
end

function process_new!(gen::Map{T,U}, args::Tuple,
                      constraints::Dict{Int,Any}, key::Int,
                      state::MapAssessState{T,U}) where {T,U}
    local subtrace::U
    kernel_args = get_args_for_key(args, key)
    if haskey(constraints, key)
        subconstraints = constraints[key]
        subtrace = assess(gen.kernel, kernel_args, subconstraints)
    else
        subtrace = assess(gen.kernel, kernel_args, EmptyAssignment())
    end
    state.num_has_choices += has_choices(subtrace) ? 1 : 0
    call = get_call_record(subtrace)
    state.score += call.score
    state.subtraces[key] = subtrace
    state.retvals[key] = call.retval
end

function assess(gen::Map{T,U}, args::Tuple, constraints::Assignment) where {T,U}
    len = length(args[1])
    nodes = collect_map_constraints(constraints, len)
    state = MapAssessState{T,U}(0., Vector{U}(undef,len), Vector{T}(undef,len), 0)
    for key=1:len
        process_new!(gen, args, nodes, key, state)
    end
    VectorTrace{T,U}(
        PersistentVector{U}(state.subtraces), PersistentVector{T}(state.retvals),
        args, state.score, len, state.num_has_choices)
end
