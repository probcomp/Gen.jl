function simulate(gen::AtDynamic{T,U,V}, args, read_trace=nothing) where {T,U,V}
    (index::V, kernel_args) = args
    subtrace::U = simulate(gen.kernel, kernel_args)
    sub_call = get_call_record(subtrace)
    call = CallRecord(sub_call.score, sub_call.retval, args)
    AtDynamicTrace(call, subtrace, index)
end
