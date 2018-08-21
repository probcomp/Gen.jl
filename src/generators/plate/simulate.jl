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

# NOTE: we're not actually specializing using the type information yet, 
# this is equivalent to a dynamic version
function codegen_simulate(gen::Type{Plate{T,U}}, args, constraints, read_trace=nothing) where {T,U}
    Core.println("generating simulate() method for plate: $gen")
    quote
        len = length(args[1])
        
        # collect initial state
        state = GenLite.PlateSimulateState(0., true, args, read_trace)
        subtraces = Vector{$U}(len)
        retvals = Vector{$T}(len)
        
        # visit each new application and run generate (or simulate) on it
        for key=1:len
            (subtraces[key], retvals[key]) = GenLite.simulate_new_trace!(gen, key, state)
        end
    
        call = CallRecord{GenLite.PersistentVector{$T}}(state.score, GenLite.PersistentVector{$T}(retvals), args)
        GenLite.VectorTrace{$T,$U}(GenLite.PersistentVector{$U}(subtraces), call, state.is_empty)
    end
end
