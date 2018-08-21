function at_dynamic_simulate(gen::AtDynamic{T,U,V}, args, read_trace=nothing) where {T,U,V}
    (index::V, kernel_args) = args
    subtrace::U = simulate(gen.kernel, kernel_args)
    sub_call = get_call_record(subtrace)
    call = CallRecord(sub_call.score, sub_call.retval, args)
    trace = AtDynamicTrace(call, subtrace, index)
    trace
end

function codegen_simulate(gen::Type{AtDynamic{T,U,V}}, args, read_trace=nothing) where {T,U,V}
    Core.println("generating simulate() method for at_dynamic: $gen")
    quote
        GenLite.at_dynamic_simulate(gen, args, read_trace)
    end
end
