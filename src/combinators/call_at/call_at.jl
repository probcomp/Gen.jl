struct CallAtAssignment{K,T} <: Assignment
    key::K
    subassmt::T
end

Base.isempty(assmt::CallAtAssignment) = isempty(assmt.subassmt)

function get_address_schema(::Type{T}) where {T<:CallAtAssignment}
    SingleDynamicKeyAddressSchema()
end

function get_subassmt(assmt::CallAtAssignment{K,T}, addr::K) where {K,T}
    assmt.key == addr ? assmt.subassmt : EmptyAssignment()
end

get_subassmt(assmt::CallAtAssignment, addr::Pair) = _get_subassmt(assmt, addr)
get_value(assmt::CallAtAssignment, addr::Pair) = _get_value(assmt, addr)
has_value(assmt::CallAtAssignment, addr::Pair) = _has_value(assmt, addr)
get_subassmts_shallow(assmt::CallAtAssignment) = ((assmt.key, assmt.subassmt),)
get_values_shallow(::CallAtAssignment) = ()

struct CallAtTrace{U,K}
    gen_fn::GenerativeFunction # the ChoiceAtCombinator (not the kernel)
    subtrace::U
    key::K
end

get_args(trace::CallAtTrace) = (get_args(trace.subtrace), trace.key)
get_retval(trace::CallAtTrace) = get_retval(trace.subtrace)
get_score(trace::CallAtTrace) = get_score(trace.subtrace)
get_gen_fn(trace::CallAtTrace) = trace.gen_fn

function get_assignment(trace::CallAtTrace)
    CallAtAssignment(trace.key, get_assignment(trace.subtrace))
end

struct CallAtCombinator{T,U,K} <: GenerativeFunction{T, CallAtTrace{U,K}}
    kernel::GenerativeFunction{T,U}
end

function call_at(kernel::GenerativeFunction{T,U}, ::Type{K}) where {T,U,K}
    CallAtCombinator{T,U,K}(kernel)
end

function assess(gen_fn::CallAtCombinator, args::Tuple, assmt::Assignment)
    key = args[end]
    kernel_args = args[1:end-1]
    if length(get_subassmts_shallow(assmt)) > 1 || length(get_values_shallow(assmt)) > 0
        error("Not all constraints were consumed")
    end
    subassmt = get_subassmt(assmt, key)
    assess(gen_fn.kernel, kernel_args, subassmt)
end

function propose(gen_fn::CallAtCombinator, args::Tuple)
    key = args[end]
    kernel_args = args[1:end-1]
    (subassmt, weight, retval) = propose(gen_fn.kernel, kernel_args)
    assmt = CallAtAssignment(key, subassmt)
    (assmt, weight, retval)
end

function initialize(gen_fn::CallAtCombinator{T,U,K}, args::Tuple,
                    assmt::Assignment) where {T,U,K}
    key = args[end]
    kernel_args = args[1:end-1]
    subassmt = get_subassmt(assmt, key) 
    (subtrace, weight) = initialize(gen_fn.kernel, kernel_args, subassmt)
    trace = CallAtTrace{U,K}(gen_fn, subtrace, key)
    (trace, weight)
end

function project(trace::CallAtTrace, selection::AddressSet)
    if has_internal_node(selection, trace.key)
        subselection = get_internal_node(selection, trace.key)
    else
        subselection = EmptyAddressSet()
    end
    project(trace.subtrace, subselection)
end

function force_update(args::Tuple, argdiff, trace::CallAtTrace{U,K},
                      assmt::Assignment) where {U,K}
    key = args[end]
    kernel_args = args[1:end-1]
    key_changed = (key != trace.key)
    subassmt = get_subassmt(assmt, key)
    if key_changed
        (subtrace, weight) = initialize(trace.gen_fn.kernel, kernel_args, subassmt)
        weight -= get_score(trace.subtrace)
        discard = get_assignment(trace)
        retdiff = DefaultRetDiff()
    else
        (subtrace, weight, subdiscard, retdiff) = force_update(
            kernel_args, unknownargdiff, trace.subtrace, subassmt)
        discard = CallAtAssignment(key, subdiscard)
    end
    new_trace = CallAtTrace{U,K}(trace.gen_fn, subtrace, key)
    (new_trace, weight, discard, retdiff)
end

function fix_update(args::Tuple, argdiff, trace::CallAtTrace{U,K},
                    assmt::Assignment) where {U,K}
    key = args[end]
    kernel_args = args[1:end-1]
    key_changed = (key != trace.key)
    subassmt = get_subassmt(assmt, key)
    if key_changed
        if !isempty(subassmt)
            error("Cannot constrain addresses under new key $key in fix_update")
        end
        (subtrace, weight) = initialize(trace.gen_fn.kernel, kernel_args, EmptyAssignment())
        weight -= project(trace.subtrace, EmptyAddressSet())
        retdiff = DefaultRetDiff()
        discard = EmptyAssignment()
    else
        (subtrace, weight, subdiscard, retdiff) = fix_update(
            kernel_args, unknownargdiff, trace.subtrace, subassmt)
        discard = CallAtAssignment(key, subdiscard)
    end
    new_trace = CallAtTrace{U,K}(trace.gen_fn, subtrace, key)
    (new_trace, weight, discard, retdiff)
end

function free_update(args::Tuple, argdiff, trace::CallAtTrace{U,K},
                     selection::AddressSet) where {U,K}
    key = args[end]
    kernel_args = args[1:end-1]
    key_changed = (key != trace.key)
    if key_changed
        if has_internal_node(selection, key)
            error("Cannot select addresses under new key $key in free_update")
        end
        (subtrace, weight) = initialize(trace.gen_fn.kernel, kernel_args, EmptyAssignment())
        weight -= project(trace.subtrace, EmptyAddressSet())
        retdiff = DefaultRetDiff()
    else
        if has_internal_node(selection, key)
            subselection = get_internal_node(selection, key)
        else
            subselection = EmptyAddressSet()
        end
        (subtrace, weight, retdiff) = free_update(
            kernel_args, unknownargdiff, trace.subtrace, subselection)
    end
    new_trace = CallAtTrace{U,K}(trace.gen_fn, subtrace, key)
    (new_trace, weight, retdiff)
end

function extend(args::Tuple, argdiff, trace::CallAtTrace{U,K},
                assmt::Assignment) where {U,K}
    key = args[end]
    kernel_args = args[1:end-1]
    key_changed = (key != trace.key)
    subassmt = get_subassmt(assmt, key)
    if key_changed
        error("Cannot remove address $(trace.key) in extend")
    end
    (subtrace, weight, retdiff) = extend(
        kernel_args, unknownargdiff, trace.subtrace, subassmt)
    new_trace = CallAtTrace{U,K}(trace.gen_fn, subtrace, key)
    (new_trace, weight, retdiff)
end

function backprop_trace(trace::CallAtTrace, selection::AddressSet, retval_grad)
    if has_internal_node(selection, trace.key)
        subselection = get_internal_node(selection, trace.key)
    else
        subselection = EmptyAddressSet()
    end
    (kernel_input_grads, value_subassmt, gradient_subassmt) = backprop_trace(
        trace.subtrace, subselection, retval_grad)
    input_grads = (kernel_input_grads..., nothing)
    value_assmt = CallAtAssignment(trace.key, value_subassmt)
    gradient_assmt = CallAtAssignment(trace.key, gradient_subassmt)
    (input_grads, value_assmt, gradient_assmt)
end

function backprop_params(trace::CallAtTrace, retval_grad)
    kernel_input_grads = backprop_params(trace.subtrace, retval_grad)
    (kernel_input_grads..., nothing)
end

export call_at
