# TODO optimize ChoiceAtTrace using type parameters

struct ChoiceAtTrace
    gen_fn::GenerativeFunction # the ChoiceAtCombinator (not the kernel)
    value::Any
    key::Any
    kernel_args::Tuple
    score::Float64
end

get_args(trace::ChoiceAtTrace) = (trace.kernel_args..., trace.key)
get_retval(trace::ChoiceAtTrace) = trace.value
get_score(trace::ChoiceAtTrace) = trace.score
get_gen_fn(trace::ChoiceAtTrace) = trace.gen_fn

struct ChoiceAtAssignment{T,K} <: Assignment
    key::K
    value::T
end

get_assmt(trace::ChoiceAtTrace) = ChoiceAtAssignment(trace.key, trace.value)
Base.isempty(::ChoiceAtAssignment) = false
function get_address_schema(::Type{T}) where {T<:ChoiceAtAssignment}
    SingleDynamicKeyAddressSchema()
end
get_value(assmt::ChoiceAtAssignment, addr::Pair) = _get_value(assmt, addr)
has_value(assmt::ChoiceAtAssignment, addr::Pair) = _has_value(assmt, addr)
function get_value(assmt::ChoiceAtAssignment{T,K}, addr::K) where {T,K}
    assmt.key == addr ? assmt.value : throw(KeyError(assmt, addr))
end
get_subassmts_shallow(assmt::ChoiceAtAssignment) = ()
get_values_shallow(assmt::ChoiceAtAssignment) = ((assmt.key, assmt.value),)

struct ChoiceAtCombinator{T,K} <: GenerativeFunction{T, ChoiceAtTrace}
    dist::Distribution{T}
end

function choice_at(dist::Distribution{T}, ::Type{K}) where {T,K}
    ChoiceAtCombinator{T,K}(dist)
end

function assess(gen_fn::ChoiceAtCombinator{T,K}, args::Tuple, assmt::Assignment) where {T,K}
    local key::K
    local value::T
    key = args[end]
    kernel_args = args[1:end-1]
    value = get_value(assmt, key)
    weight = logpdf(gen_fn.dist, value, kernel_args...)
    (weight, value)
end

function propose(gen_fn::ChoiceAtCombinator{T,K}, args::Tuple) where {T,K}
    local key::K
    local value::T
    key = args[end]
    kernel_args = args[1:end-1]
    value = random(gen_fn.dist, kernel_args...)
    score = logpdf(gen_fn.dist, value, kernel_args...)
    assmt = ChoiceAtAssignment(key, value)
    (assmt, score, value)
end

function initialize(gen_fn::ChoiceAtCombinator{T,K}, args::Tuple, assmt::Assignment) where {T,K}
    local key::K
    local value::T
    key = args[end]
    kernel_args = args[1:end-1]
    constrained = has_value(assmt, key)
    value = constrained ? get_value(assmt, key) : random(gen_fn.dist, kernel_args...)
    score = logpdf(gen_fn.dist, value, kernel_args...)
    trace = ChoiceAtTrace(gen_fn, value, key, kernel_args, score)
    weight = constrained ? score : 0.
    (trace, weight)
end

function project(trace::ChoiceAtTrace, selection::AddressSet)
    has_leaf_node(selection, trace.key) ? trace.score : 0.
end

function force_update(args::Tuple, argdiff, trace::ChoiceAtTrace,
                      assmt::Assignment)
    key = args[end]
    kernel_args = args[1:end-1]
    key_changed = (key != trace.key)
    constrained = has_value(assmt, key)
    if key_changed && constrained
        new_value = get_value(assmt, key)
        discard = ChoiceAtAssignment(trace.key, trace.value)
    elseif !key_changed && constrained
        new_value = get_value(assmt, key)
        discard = ChoiceAtAssignment(key, trace.value)
    elseif !key_changed && !constrained
        new_value = trace.value
        discard = EmptyAssignment()
    else
        error("New address $key not constrained in force_update")
    end
    new_score = logpdf(trace.gen_fn.dist, new_value, kernel_args...)
    new_trace = ChoiceAtTrace(trace.gen_fn, new_value, key, kernel_args, new_score)
    weight = new_score - trace.score
    (new_trace, weight, discard, DefaultRetDiff())
end

function fix_update(args::Tuple, argdiff, trace::ChoiceAtTrace,
                    assmt::Assignment)
    key = args[end]
    kernel_args = args[1:end-1]
    key_changed = (key != trace.key)
    constrained = has_value(assmt, key)
    if !key_changed && constrained 
        new_value = get_value(assmt, key)
    elseif !key_changed && !constrained
        new_value = trace.value
    elseif key_changed && !constrained
        new_value = random(trace.gen_fn.dist, kernel_args...)
    else
        error("Cannot constrain new address $key in fix_update")
    end
    new_score = logpdf(trace.gen_fn.dist, new_value, kernel_args...)
    if !key_changed && constrained 
        weight = new_score - trace.score
        discard = ChoiceAtAssignment(key, trace.value)
    elseif !key_changed && !constrained
        weight = new_score - trace.score
        discard = EmptyAssignment()
    elseif key_changed && !constrained
        weight = 0.
        discard = EmptyAssignment()
    end
    new_trace = ChoiceAtTrace(trace.gen_fn, new_value, key, kernel_args, new_score)
    (new_trace, weight, discard, DefaultRetDiff())
end

function free_update(args::Tuple, argdiff, trace::ChoiceAtTrace,
                     selection::AddressSet)
    key = args[end]
    kernel_args = args[1:end-1]
    key_changed = (key != trace.key)
    selected = has_leaf_node(selection, key)
    if !key_changed && selected 
        new_value = random(trace.gen_fn.dist, kernel_args...)
    elseif !key_changed && !selected
        new_value = trace.value
    elseif key_changed && !selected
        new_value = random(trace.gen_fn.dist, kernel_args...)
    else
        error("Cannot select new address $key in free_update")
    end
    new_score = logpdf(trace.gen_fn.dist, new_value, kernel_args...)
    if !key_changed && selected 
        weight = 0.
    elseif !key_changed && !selected
        weight = new_score - trace.score
    elseif key_changed && !selected
        weight = 0.
    end
    new_trace = ChoiceAtTrace(trace.gen_fn, new_value, key, kernel_args, new_score)
    (new_trace, weight, DefaultRetDiff())
end

function extend(args::Tuple, argdiff, trace::ChoiceAtTrace,
                assmt::Assignment)
    key = args[end]
    kernel_args = args[1:end-1]
    key_changed = (key != trace.key)
    if key_changed
        error("Cannot remove address $(trace.key) in extend")
    end
    constrained = has_value(assmt, key)
    if constrained
        error("Cannot change value of address $key in extend")
    end
    if (length(collect(get_values_shallow(assmt))) > 0 ||
        length(collect(get_subassmts_shallow(assmt))) > 0)
        error("Cannot constrain addresses that do not exist")
    end
    new_score = logpdf(trace.gen_fn.dist, trace.value, kernel_args...)
    new_trace = ChoiceAtTrace(trace.gen_fn, trace.value, key, kernel_args, new_score)
    weight = new_score - trace.score
    (new_trace, weight, DefaultRetDiff())
end

function backprop_trace(trace::ChoiceAtTrace, selection::AddressSet, retval_grad)
    kernel_arg_grads = logpdf_grad(trace.gen_fn.dist, trace.value, trace.kernel_args...)
    if has_leaf_node(selection, trace.key)
        value_assmt = ChoiceAtAssignment(trace.key, trace.value)
        choice_grad = kernel_arg_grads[1]
        if retval_grad != nothing
            choice_grad += retval_grad
        end
        gradient_assmt = ChoiceAtAssignment(trace.key, choice_grad)
    else
        value_assmt = EmptyAssignment()
        gradient_assmt = EmptyAssignment()
    end
    input_grads = (kernel_arg_grads[2:end]..., nothing)
    (input_grads, value_assmt, gradient_assmt)
end

function backprop_params(trace::ChoiceAtTrace, retval_grad)
    kernel_arg_grads = logpdf_grad(trace.gen_fn.dist, trace.value, trace.kernel_args...)
    (kernel_arg_grads[2:end]..., nothing)
end

export choice_at
