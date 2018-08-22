mutable struct PlateSimulateState{U,X}
    score::Float64
    is_empty::Bool
    args::U
    read_trace::X
end

function simulate_new_trace!(gen::Plate{T,U}, key::Int, state::PlateSimulateState) where {T,U}
    kernel_args = get_args_for_key(state.args, key)
    subtrace::U = simulate(gen.kernel, kernel_args, state.read_trace)
    state.is_empty = state.is_empty && !has_choices(subtrace)
    call::CallRecord = get_call_record(subtrace)
    state.score += call.score
    (subtrace, call.retval::T)
end

function simulate(gen::Plate{T,U}, args, constraints, read_trace=nothing) where {T,U}
    len = length(args[1])
    
    # collect initial state
    state = Gen.PlateSimulateState(0., true, args, read_trace)
    subtraces = Vector{U}(undef, len)
    retvals = Vector{T}(undef, len)
    
    # visit each new application and run generate (or simulate) on it
    for key=1:len
        (subtraces[key], retvals[key]) = Gen.simulate_new_trace!(gen, key, state)
    end

    call = CallRecord{Gen.PersistentVector{T}}(state.score, Gen.PersistentVector{T}(retvals), args)
    Gen.VectorTrace{T,U}(Gen.PersistentVector{U}(subtraces), call, state.is_empty)
end
