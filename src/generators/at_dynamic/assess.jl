function assess(gen::AtDynamic{T,U,V}, args, constraints, read_trace=nothing) where {T,U,V}
    (index::V, kernel_args) = args
    if length(get_internal_nodes(constraints)) + length(get_leaf_nodes(constraints)) > 1
        error("Not all constraints were consumed")
    end
    kernel_constraints = get_internal_node(constraints, index)
    subtrace::U = assess(gen.kernel, kernel_args, kernel_constraints, read_trace)
    sub_call = get_call_record(subtrace)
    call = CallRecord(sub_call.score, sub_call.retval, args)
    trace = AtDynamicTrace(call, subtrace, index)
    trace
end
