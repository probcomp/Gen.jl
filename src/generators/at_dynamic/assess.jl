function assess(gen::AtDynamic{T,U,K}, args, constraints, read_trace=nothing) where {T,U,K}
    (key::K, kernel_args) = args
    if length(get_internal_nodes(constraints)) + length(get_leaf_nodes(constraints)) > 1
        error("Not all constraints were consumed")
    end
    kernel_constraints = get_internal_node(constraints, key)
    subtrace::U = assess(gen.kernel, kernel_args, kernel_constraints, read_trace)
    sub_call = get_call_record(subtrace)
    call = CallRecord{T}(sub_call.score, sub_call.retval, args)
    trace = AtDynamicTrace{T,U,K}(call, subtrace, key)
    trace
end
