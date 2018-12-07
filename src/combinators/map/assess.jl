mutable struct MapAssessState{T,U}
    weight::Float64
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
        (weight, retval) = assess(gen.kernel, kernel_args, subconstraints)
    else
        (weight, retval) = assess(gen.kernel, kernel_args, EmptyAssignment())
    end
    state.num_has_choices += has_choices(subtrace) ? 1 : 0
    state.weight += weight
    state.retvals[key] = retval
end

function assess(gen::Map{T,U}, args::Tuple, constraints::Assignment) where {T,U}
    len = length(args[1])
    nodes = collect_map_constraints(constraints, len)
    state = MapAssessState{T,U}(0., Vector{U}(undef,len), Vector{T}(undef,len), 0)
    for key=1:len
        process_new!(gen, args, nodes, key, state)
    end
    (state.weight, state.retvals)
end
