function simulate(gen::AtDynamic{T,U,K}, args) where {T,U,K}
    (key::K, kernel_args) = args
    subtrace::U = simulate(gen.kernel, kernel_args)
    sub_call = get_call_record(subtrace)
    call = CallRecord{T}(sub_call.score, sub_call.retval, args)
    AtDynamicTrace{T,U,K}(call, subtrace, key)
end

function initialize(gen::AtDynamic{T,U,K}, args, constraints) where {T,U,K}
    (key::K, kernel_args) = args
    if has_internal_node(constraints, key)
        subconstraints = get_internal_node(constraints, key)
    else 
        subconstraints = EmptyAssignment()
    end
    (subtrace::U, weight) = initialize(gen.kernel, kernel_args, subconstraints)
    sub_call = get_call_record(subtrace)
    call = CallRecord{T}(sub_call.score, sub_call.retval, args)
    trace = AtDynamicTrace{T,U,K}(call, subtrace, key)
    (trace, weight)
end
