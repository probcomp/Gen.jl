function simulate(gen::AtDynamic{T,U,K}, args, read_trace=nothing) where {T,U,K}
    (key::K, kernel_args) = args
    subtrace::U = simulate(gen.kernel, kernel_args)
    sub_call = get_call_record(subtrace)
    call = CallRecord{T}(sub_call.score, sub_call.retval, args)
    AtDynamicTrace{T,U,K}(call, subtrace, key)
end
