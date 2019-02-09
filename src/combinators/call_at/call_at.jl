struct CallAtChoiceMap{K,T} <: ChoiceMap
    key::K
    submap::T
end

Base.isempty(choices::CallAtChoiceMap) = isempty(choices.submap)

function get_address_schema(::Type{T}) where {T<:CallAtChoiceMap}
    SingleDynamicKeyAddressSchema()
end

function get_submap(choices::CallAtChoiceMap{K,T}, addr::K) where {K,T}
    choices.key == addr ? choices.submap : EmptyChoiceMap()
end

get_submap(choices::CallAtChoiceMap, addr::Pair) = _get_submap(choices, addr)
get_value(choices::CallAtChoiceMap, addr::Pair) = _get_value(choices, addr)
has_value(choices::CallAtChoiceMap, addr::Pair) = _has_value(choices, addr)
get_submaps_shallow(choices::CallAtChoiceMap) = ((choices.key, choices.submap),)
get_values_shallow(::CallAtChoiceMap) = ()

# TODO optimize CallAtTrace using type parameters

struct CallAtTrace <: Trace
    gen_fn::GenerativeFunction # the ChoiceAtCombinator (not the kernel)
    subtrace::Any
    key::Any
end

get_args(trace::CallAtTrace) = (get_args(trace.subtrace), trace.key)
get_retval(trace::CallAtTrace) = get_retval(trace.subtrace)
get_score(trace::CallAtTrace) = get_score(trace.subtrace)
get_gen_fn(trace::CallAtTrace) = trace.gen_fn

function get_choices(trace::CallAtTrace)
    CallAtChoiceMap(trace.key, get_choices(trace.subtrace))
end

struct CallAtCombinator{T,U,K} <: GenerativeFunction{T, CallAtTrace}
    kernel::GenerativeFunction{T,U}
end

function call_at(kernel::GenerativeFunction{T,U}, ::Type{K}) where {T,U,K}
    CallAtCombinator{T,U,K}(kernel)
end

function accepts_output_grad(gen_fn::CallAtCombinator)
    accepts_output_grad(gen_fn.kernel)
end

unpack_call_at_args(args) = (args[end], args[1:end-1])


function assess(gen_fn::CallAtCombinator, args::Tuple, choices::ChoiceMap)
    (key, kernel_args) = unpack_call_at_args(args)
    if length(get_submaps_shallow(choices)) > 1 || length(get_values_shallow(choices)) > 0
        error("Not all constraints were consumed")
    end
    submap = get_submap(choices, key)
    assess(gen_fn.kernel, kernel_args, submap)
end

function propose(gen_fn::CallAtCombinator, args::Tuple)
    (key, kernel_args) = unpack_call_at_args(args)
    (submap, weight, retval) = propose(gen_fn.kernel, kernel_args)
    choices = CallAtChoiceMap(key, submap)
    (choices, weight, retval)
end

function simulate(gen_fn::CallAtCombinator, args::Tuple)
    (key, kernel_args) = unpack_call_at_args(args)
    subtrace = simulate(gen_fn.kernel, kernel_args)
    CallAtTrace(gen_fn, subtrace, key)
end

function generate(gen_fn::CallAtCombinator{T,U,K}, args::Tuple,
                    choices::ChoiceMap) where {T,U,K}
    (key, kernel_args) = unpack_call_at_args(args)
    submap = get_submap(choices, key) 
    (subtrace, weight) = generate(gen_fn.kernel, kernel_args, submap)
    trace = CallAtTrace(gen_fn, subtrace, key)
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

function update(trace::CallAtTrace, args::Tuple, argdiff,
                choices::ChoiceMap)
    (key, kernel_args) = unpack_call_at_args(args)
    key_changed = (key != trace.key)
    submap = get_submap(choices, key)
    if key_changed
        (subtrace, weight) = generate(trace.gen_fn.kernel, kernel_args, submap)
        weight -= get_score(trace.subtrace)
        discard = get_choices(trace)
        retdiff = DefaultRetDiff()
    else
        (subtrace, weight, retdiff, subdiscard) = update(
            trace.subtrace, kernel_args, unknownargdiff, submap)
        discard = CallAtChoiceMap(key, subdiscard)
    end
    new_trace = CallAtTrace(trace.gen_fn, subtrace, key)
    (new_trace, weight, retdiff, discard)
end

function regenerate(trace::CallAtTrace, args::Tuple, argdiff,
                    selection::AddressSet)
    (key, kernel_args) = unpack_call_at_args(args)
    key_changed = (key != trace.key)
    if key_changed
        if has_internal_node(selection, key)
            error("Cannot select addresses under new key $key in regenerate")
        end
        (subtrace, weight) = generate(trace.gen_fn.kernel, kernel_args, EmptyChoiceMap())
        weight -= project(trace.subtrace, EmptyAddressSet())
        retdiff = DefaultRetDiff()
    else
        if has_internal_node(selection, key)
            subselection = get_internal_node(selection, key)
        else
            subselection = EmptyAddressSet()
        end
        (subtrace, weight, retdiff) = regenerate(
            trace.subtrace, kernel_args, unknownargdiff, subselection)
    end
    new_trace = CallAtTrace(trace.gen_fn, subtrace, key)
    (new_trace, weight, retdiff)
end

function extend(trace::CallAtTrace, args::Tuple, argdiff,
                choices::ChoiceMap)
    (key, kernel_args) = unpack_call_at_args(args)
    key_changed = (key != trace.key)
    submap = get_submap(choices, key)
    if key_changed
        error("Cannot remove address $(trace.key) in extend")
    end
    (subtrace, weight, retdiff) = extend(
        trace.subtrace, kernel_args, unknownargdiff, submap)
    new_trace = CallAtTrace(trace.gen_fn, subtrace, key)
    (new_trace, weight, retdiff)
end

function choice_gradients(trace::CallAtTrace, selection::AddressSet, retval_grad)
    if has_internal_node(selection, trace.key)
        subselection = get_internal_node(selection, trace.key)
    else
        subselection = EmptyAddressSet()
    end
    (kernel_input_grads, value_submap, gradient_submap) = choice_gradients(
        trace.subtrace, subselection, retval_grad)
    input_grads = (kernel_input_grads..., nothing)
    value_choices = CallAtChoiceMap(trace.key, value_submap)
    gradient_choices = CallAtChoiceMap(trace.key, gradient_submap)
    (input_grads, value_choices, gradient_choices)
end

function accumulate_param_gradients!(trace::CallAtTrace, retval_grad)
    kernel_input_grads = accumulate_param_gradients!(trace.subtrace, retval_grad)
    (kernel_input_grads..., nothing)
end

export call_at
