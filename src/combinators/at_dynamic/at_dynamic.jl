# TODO write a version that puts a distribution call at a specific address
# DistAtCombinator
# GenFnAtCombinator

# could we use a DynamicGenFunction?
# yes, it would not be fast
# is that okay?

struct AtTrace{U,K}
    gen_fn::GenerativeFunction
    subtrace::U
    key::K
end

get_args(trace::AtTrace) = (get_args(trace.subtrace), trace.key)
get_retval(trace::AtTrace) = get_retval(trace.subtrace)
get_score(trace::AtTrace) = get_score(trace.subtrace)
get_gen_fn(trace::AtTrace) = trace.gen_fn

struct AtTraceAssignment{U,K} <: Assignment
    trace::AtTrace{U,K}
end

get_assignment(trace::AtTrace) = AtTraceAssignment(trace)

Base.isempty(assmt::AtTraceAssignment) = isempty(get_assignment(assmt.trace.subtrace))
get_address_schema(::Type{AtTraceAssignment}) = SingleDynamicKeyAddressSchema()

function get_subassmt(assmt::AtTraceAssignment{U,K}, addr::K) where {U,K}
    if assmt.trace.key == addr
        get_assignment(assmt.trace.subtrace)
    else
        throw(KeyError(assmt, addr))
    end
end

get_subassmt(assmt::AtTraceAssignment, addr::Pair) = _get_subassmt(assmt, addr)
get_value(assmt::AtTraceAssignment, addr::Pair) = _get_value(assmt, addr)
has_value(assmt::AtTraceAssignment, addr::Pair) = _has_value(assmt, addr)

function get_subassmts_shallow(assmt::AtTraceAssignment)
    ((assmt.trace.key, get_assignment(assmt.trace.subtrace)),)
end

get_values_shallow(::AtTraceAssignment) = ()

 struct AtCombinator{T,U,K} <: GenerativeFunction{T, AtTrace{U,K}}
    kernel::GenerativeFunction{T,U}
end

function at_combinator(kernel::GenerativeFunction{T,U}, ::Type{K}) where {T,U,K}
    AtCombinator{T,U,K}(kernel)
end

function assess(gen_fn::AtCombinator, args::Tuple, assmt::Assignment)
    key = args[1]
    kernel_args = args[2:end]
    if length(get_subassmts_shallow(assmt)) > 1 || length(get_values_shallow(assmt)) > 0
        error("Not all constraints were consumed")
    end
    subassmt = get_subassmt(assmt, key)
    assess(gen_fn.kernel, kernel_args, subassmt)
end

struct AtAssignment{K,T} <: Assignment
    key::K
    subassmt::T
end

Base.isempty(assmt::AtAssignment) = isempty(assmt.subassmt)

function get_address_schema(::Type{T}) where {T<:AtAssignment}
    SingleDynamicKeyAddressSchema()
end

function get_subassmt(assmt::AtAssignment{U,K}, addr::K) where {U,K}
    assmt.key == addr ? assmt.subassmt : throw(KeyError(assmt, addr))
end

get_subassmt(assmt::AtAssignment, addr::Pair) = _get_subassmt(assmt, addr)
get_value(assmt::AtAssignment, addr::Pair) = _get_value(assmt, addr)
has_value(assmt::AtAssignment, addr::Pair) = _has_value(assmt, addr)

get_subassmts_shallow(assmt::AtAssignment) = ((assmt.key, assmt.subassmt),)
get_values_shallow(::AtAssignment) = ()


function propose(gen_fn::AtCombinator, args::Tuple)
    key = args[end]
    kernel_args = args[1:end-1]
    (subassmt, weight, retval) = propose(gen_fn.kernel, kernel_args)
    assmt = AtAssignment(key, subassmt)
    (assmt, weight, retval)
end

function initialize(gen_fn::AtCombinator{T,U,K}, args::Tuple, assmt::Assignment) where {T,U,K}
    println("args: $args")
    key = args[end]
    kernel_args = args[1:end-1]
    subassmt = get_subassmt(assmt, key) 
    (subtrace, weight) = initialize(gen_fn.kernel, kernel_args, subassmt)
    trace = AtTrace{U,K}(gen_fn, subtrace, key)
    (trace, weight)
end

function project(trace::AtTrace, selection::AddressSet)
    if has_internal_node(selection, trace.key)
        subselection = get_internal_node(selection, trace.key)
    else
        subselection = EmptyAddressSet()
    end
    project(trace.subtrace, subselection)
end

# TODO implement other argdiffs
function force_update(args::Tuple, ::NoArgDiff, trace::AtTrace{U,K}, assmt::Assignment) where {U,K}
    key = args[end]
    kernel_args = args[1:end-1]
    subassmt = get_subassmt(assmt, key)
    (new_subtrace, weight, subdiscard, retdiff) = force_update(
        kernel_args, noargdiff, subtrace, subassmt)
    discard = AtAssignment(key, subdiscard)
    new_trace = AtTrace{U,K}(trace.gen_fn, subtrace, key)
    (new_trace, weight, discard, retdiff)
end

function force_update(args::Tuple, ::UnknownArgDiff, trace::AtTrace{U,K}, assmt::Assignment) where {U,K}
    key = args[end]
    kernel_args = args[1:end-1]
    prev_args = get_args(trace)
    prev_key = prev_args[end]
    prev_kernel_args = prev_args[1:end-1]
    if prev_key == new_key
    end
    subassmt = get_subassmt(assmt, key)
    (new_subtrace, weight, subdiscard, retdiff) = force_update(
        kernel_args, noargdiff, subtrace, subassmt)
    discard = AtAssignment(key, subdiscard)
    new_trace = AtTrace(trace.gen_fn, subtrace, key)
    (new_trace, weight, discard, retdiff)
end


# TODO implement other argdiffs
function fix_update(args::Tuple, ::NoArgDiff, trace::AtTrace{U,K}, assmt::Assignment) where {U,K}
    key = args[end]
    kernel_args = args[1:end-1]
    subassmt = get_subassmt(assmt, key)
    (new_subtrace, weight, subdiscard, retdiff) = fix_update(
        kernel_args, noargdiff, subtrace, subassmt)
    discard = AtAssignment(key, subdiscard)
    new_trace = AtTrace{U,K}(trace.gen_fn, subtrace, key)
    (new_trace, weight, discard, retdiff)
end

# TODO implement other argdiffs
function free_update(args::Tuple, ::NoArgDiff, trace::AtTrace{U,K}, selection::AddressSet) where {U,K}
    key = args[end]
    kernel_args = args[1:end-1]
    if has_internal_node(selection, key)
        subselection = get_internal_node(selection, key)
    else
        subselection = EmptyAddressSet()
    end
    (new_subtrace, weight, retdiff) = free_update(
        kernel_args, noargdiff, subtrace, subselection)
    new_trace = AtTrace{U,K}(trace.gen_fn, subtrace, key)
    (new_trace, weight, retdiff)
end

# TODO implement other argdiffs
function extend(args::Tuple, ::NoArgDiff, trace::AtTrace{U,K}, assmt::Assignment) where {U,K}
    key = args[end]
    kernel_args = args[1:end-1]
    subassmt = get_subassmt(assmt, key)
    (new_subtrace, weight, retdiff) = extend(
        kernel_args, noargdiff, subtrace, subassmt)
    new_trace = AtTrace{U,K}(trace.gen_fn, subtrace, key)
    (new_trace, weight, retdiff)
end

function backprop_trace(trace::AtTrace, selection::AddressSet, retval_grad)
    (kernel_input_grads, value_subassmt, gradient_subassmt) = backprop_trace(
        subtrace, subselection, retval_grad)
    input_grads = (nothing, kernel_input_grads...)
    value_assmt = AtAssignment(trace.key, value_subassmt)
    gradient_assmt = AtAssignment(trace.key, gradient_subassmt)
    (input_grads, value_assmt, gradient_assmt)
end

function backprop_params(trace::AtTrace, retval_grad)
    kernel_input_grads = backprop_params(subtrace, retval_grad)
    (nothing, kernel_input_grads...)
end

export at_combinator
