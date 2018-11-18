mutable struct PlateAssessState{T,U}
    score::Float64
    subtraces::Vector{U}
    retvals::Vector{T}
    num_has_choices::Bool
end

function process_new!(gen::Plate{T,U}, args::Tuple,
                      constraints::Dict{Int,Any}, key::Int,
                      state::PlateAssessState{T,U}) where {T,U}
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

function assess(gen::Plate{T,U}, args::Tuple, constraints::Assignment) where {T,U}
    len = length(args[1])
    nodes = collect_plate_constraints(constraints, len)
    state = PlateAssessState{T,U}(0., Vector{U}(undef,len), Vector{T}(undef,len), 0)
    for key=1:len
        process_new!(gen, args, nodes, key, state)
    end
    VectorTrace{T,U}(
        PersistentVector{U}(subtraces), PersistentVector{T}(retvals),
        args, len, state.num_has_choices)
end
