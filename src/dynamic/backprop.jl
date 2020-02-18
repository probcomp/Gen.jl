import Zygote

###############################
# accumulate_param_gradients! #
###############################

struct ParamStore
    params::Dict{Any,Any}
end

import Base: +

Zygote.@adjoint ParamStore(params) = ParamStore(params), store_grad -> (nothing,)

function +(a::ParamStore, b::ParamStore)
    params = Dict()
    for (k, v) in Iterators.flatten((a.params, b.params))
        if !haskey(params, k)
            params[k] = v
        else
            params[k] += v
        end
    end
    ParamStore(params)
end


mutable struct GFBackpropParamsState
    args::Tuple
    trace::DynamicDSLTrace
    visitor::AddressVisitor
    scaler::Float64
    params::ParamStore
    score::Float64
    cur_gen_fn::DynamicDSLFunction
end

function read_param(state::GFBackpropParamsState, name::Symbol)
    read_param(state, state.params, name)
end

function read_param(
        state::GFBackpropParamsState, params::ParamStore, name::Symbol)
    state.cur_gen_fn.params[name]
end


Zygote.@adjoint read_param(state::GFBackpropParamsState, params::ParamStore, name::Symbol) = begin
    retval = read_param(state, params, name)
    cur_gen_fn = state.cur_gen_fn
    fn = (param_grad) -> begin
        state_grad = nothing
        params_grad = ParamStore(Dict{Any,Any}((cur_gen_fn, name) => param_grad))
        (state_grad, params_grad, nothing)
    end
    (retval, fn)
end

function GFBackpropParamsState(args, trace::DynamicDSLTrace, param_store, scaler)
    score = 0.
    visitor = AddressVisitor()
    GFBackpropParamsState(args, trace, visitor, scaler, param_store, score, get_gen_fn(trace))
end

function traceat(
        state::GFBackpropParamsState, dist::Distribution{T}, args, key) where {T}
    visit!(state.visitor, key)
    retval::T = get_choice(state.trace, key).retval
    state.score += logpdf(dist, retval, args...)
    retval
end

pretend_call_param_gradients(subtrace, selection, args) = get_retval(subtrace)

Zygote.@adjoint pretend_call_param_gradients(subtrace, args, scaler) = begin
    retval = pretend_call_param_gradients(subtrace, args, scaler)
    fn = (retval_grad) -> begin
        arg_grads = accumulate_param_gradients!(subtrace, retval_grad, scaler)
        (nothing, arg_grads, nothing)
    end
    (retval, fn)
end

function traceat(
        state::GFBackpropParamsState, gen_fn::GenerativeFunction{T,U}, args, key) where {T,U}
    visit!(state.visitor, key)
    subtrace = get_call(state.trace, key).subtrace
    get_gen_fn(subtrace) === gen_fn || gen_fn_changed_error(key)
    pretend_call_param_gradients(subtrace, args, state.scaler)
end

function splice(
        state::GFBackpropParamsState, gen_fn::DynamicDSLFunction, args::Tuple)
    prev_gen_fn = state.cur_gen_fn
    state.cur_gen_fn = gen_fn
    retval = exec(gen_fn, state, args)
    state.cur_gen_fn = prev_gen_fn
    retval 
end

function accumulate_param_gradients!(trace::DynamicDSLTrace, retval_grad, scaler=1.)
    gen_fn = trace.gen_fn

    fn = (args, param_store) -> begin
        state = GFBackpropParamsState(args, trace, param_store, scaler)
        retval = exec(gen_fn, state, args)
        (state.score, retval)
    end

    dummy_param_store = ParamStore(Dict{Any,Any}())
    _, back = Zygote.pullback(fn, get_args(trace), dummy_param_store)
    arg_grads, param_store_grad = back((1., retval_grad))
    
    for ((gen_fn, name), grad) in param_store_grad.params
        gen_fn.params_grad[name] += grad * scaler
    end

    arg_grads
end


####################
# choice_gradients #
####################

mutable struct GFBackpropTraceState
    trace::DynamicDSLTrace
    score::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
    selection::Selection
end

function GFBackpropTraceState(trace, params, selection)
    score = 0.
    visitor = AddressVisitor()
    GFBackpropTraceState(trace, score, visitor, params, selection)
end

function get_addr(addr_so_far::Vector, key)
    push!(addr_so_far, key)
    addr = foldr(=>, addr_so_far)
    pop!(addr_so_far)
    addr
end

function get_selected(
        trie::Trie, grad_trie::NamedTuple, selection::Selection, addr_so_far)

    values = choicemap()
    grads = choicemap()

    for (key, record) in get_leaf_nodes(trie)
        if record.is_choice
            if get_addr(addr_so_far, key) in selection
                values[key] = record.subtrace_or_retval
                grads[key] = grad_trie.leaf_nodes[key].subtrace_or_retval
            end
        elseif haskey(grad_trie.leaf_nodes, key)
            (choice_vals, choice_grads) = grad_trie.leaf_nodes[key].subtrace_or_retval
            set_submap!(values, key, choice_vals)
            set_submap!(grads, key, choice_grads)
        end
    end

    for (key, subtrie) in get_internal_nodes(trie)
        grad_subtrie = grad_trie.internal_nodes[key]
        push!(addr_so_far, key)
        values_submap, grads_submap = get_selected(
                subtrie, grad_subtrie, selection, addr_so_far)
        pop!(addr_so_far)
        set_submap!(values, key, values_submap)
        set_submap!(grads, key, grads_submap)
    end

    (values, grads)
end

function traceat(
        state::GFBackpropTraceState, dist::Distribution{T}, args, key) where {T}
    visit!(state.visitor, key)
    retval::T = get_choice(state.trace, key).retval
    state.score += logpdf(dist, retval, args...)
    retval
end

pretend_call_choice_gradients(subtrace, selection, args) = get_retval(subtrace)

Zygote.@adjoint pretend_call_choice_gradients(subtrace, selection, args) = begin
    retval = pretend_call_choice_gradients(subtrace, selection, args)
    fn = (retval_grad) -> begin
        (arg_grads, choice_vals, choice_grads) = choice_gradients(subtrace, selection, retval_grad)
        # NOTE: we are using (choice_vals, choice_grads) as the adjoint for the subtrace
        ((choice_vals, choice_grads), nothing, arg_grads)
    end
    (retval, fn)
end

function traceat(
        state::GFBackpropTraceState, gen_fn::GenerativeFunction{T,U},
        args, key) where {T,U}
    visit!(state.visitor, key)
    subtrace = get_call(state.trace, key).subtrace
    get_gen_fn(subtrace) === gen_fn || gen_fn_changed_error(key)
    pretend_call_choice_gradients(subtrace, state.selection[key], args)
end

function splice(
        state::GFBackpropTraceState, gen_fn::DynamicDSLFunction, args::Tuple)
    prev_params = state.params
    state.params = gen_fn.params
    retval = exec(gen_fn, state, args)
    state.params = prev_params
    retval
end

function choice_gradients(
        trace::DynamicDSLTrace, selection::Selection, retval_grad)
    gen_fn = trace.gen_fn

    fn = (args, trace) -> begin
        state = GFBackpropTraceState(trace, gen_fn.params, selection)
        retval = exec(gen_fn, state, args)
        (state.score, retval)
    end

    _, back = Zygote.pullback(fn, get_args(trace), trace)
    arg_grads, trace_grad_ref = back((1., retval_grad))
    grad_trie = trace_grad_ref[].trie

    choice_vals, choice_grads = get_selected(
        trace.trie, grad_trie, selection, [])

    (arg_grads, choice_vals, choice_grads)
end
