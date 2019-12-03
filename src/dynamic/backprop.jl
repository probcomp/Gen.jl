import Zygote

function maybe_track(arg, has_argument_grad::Bool, tape)
    has_argument_grad ? track(arg, tape) : arg
end

@noinline function ReverseDiff.special_reverse_exec!(
        instruction::ReverseDiff.SpecialInstruction{D}) where {D <: Distribution}
    dist::D = instruction.func
    args_maybe_tracked = instruction.input
    score_tracked = instruction.output
    arg_grads = logpdf_grad(dist, map(value, args_maybe_tracked)...)
    value_tracked = args_maybe_tracked[1]
    value_grad = arg_grads[1]
    if istracked(value_tracked)
        if has_output_grad(dist)
            increment_deriv!(value_tracked, value_grad * deriv(score_tracked))
        else
            error("Gradient required but not available for return value of distribution $dist")
        end
    end
    for (i, (arg_maybe_tracked, grad, has_grad)) in enumerate(
            zip(args_maybe_tracked[2:end], arg_grads[2:end], has_argument_grads(dist)))
        if istracked(arg_maybe_tracked)
            if has_grad
                increment_deriv!(arg_maybe_tracked, grad * deriv(score_tracked))
            else
                error("Gradient required but not available for argument $i of $dist")
            end
        end
    end
    nothing
end


###################
# accumulate_param_gradients! #
###################

mutable struct GFBackpropParamsState
    trace::DynamicDSLTrace
    score::TrackedReal
    tape::InstructionTape
    visitor::AddressVisitor
    scaler::Float64

    # only those tracked parameters that are in scope (not including splice calls)
    tracked_params::Dict{Symbol,Any}

    # tracked parameters for all (nested) splice calls
    splice_tracked_params::Dict{Tuple{GenerativeFunction,Symbol},Any}
end

function track_params(tape, params)
    tracked_params = Dict{Symbol,Any}()
    for (name, value) in params
        tracked_params[name] = track(value, tape)
    end
    tracked_params
end

function GFBackpropParamsState(trace::DynamicDSLTrace, tape, params, scaler)
    tracked_params = track_params(tape, params)
    splice_tracked_params = Dict{Tuple{GenerativeFunction,Symbol},Any}()
    score = track(0., tape)
    GFBackpropParamsState(trace, score, tape, AddressVisitor(), scaler,
        tracked_params, splice_tracked_params)
end

function read_param(state::GFBackpropParamsState, name::Symbol)
    value = state.tracked_params[name]
    value
end

function traceat(state::GFBackpropParamsState, dist::Distribution{T},
              args_maybe_tracked, key) where {T}
    local retval::T
    visit!(state.visitor, key)
    retval = get_choice(state.trace, key).retval
    args = map(value, args_maybe_tracked)
    score_tracked = track(logpdf(dist, retval, args...), state.tape)
    record!(state.tape, ReverseDiff.SpecialInstruction, dist,
        (retval, args_maybe_tracked...,), score_tracked)
    state.score += score_tracked
    retval
end

struct BackpropParamsRecord
    gen_fn::GenerativeFunction
    subtrace::Any
    scaler::Float64
end

function traceat(state::GFBackpropParamsState, gen_fn::GenerativeFunction{T},
              args_maybe_tracked, key) where {T}
    local retval::T
    visit!(state.visitor, key)
    subtrace = get_call(state.trace, key).subtrace
    retval = get_retval(subtrace)
    if accepts_output_grad(gen_fn)
        retval_maybe_tracked = track(retval, state.tape)
        @assert istracked(retval_maybe_tracked)
    else
        retval_maybe_tracked = retval
        @assert !istracked(retval_maybe_tracked)
    end
    record!(state.tape, ReverseDiff.SpecialInstruction,
        BackpropParamsRecord(gen_fn, subtrace, state.scaler),
        (args_maybe_tracked...,), retval_maybe_tracked)
    retval_maybe_tracked 
end

function splice(state::GFBackpropParamsState, gen_fn::DynamicDSLFunction,
                args_maybe_tracked::Tuple)

    # save previous tracked parameter scope
    prev_tracked_params = state.tracked_params
    
    # construct new tracked parameter scope
    state.tracked_params = Dict{Symbol,Any}()
    for name in keys(gen_fn.params)
        if haskey(state.splice_tracked_params, (gen_fn, name))
            # parameter was already tracked in another splice call
            state.tracked_params[name] = state.splice_tracked_params[(gen_fn, name)]
        else
            # parameter was not already tracked
            tracked = track(get_param(gen_fn, name), state.tape)
            state.tracked_params[name] = tracked
            state.splice_tracked_params[(gen_fn, name)] = tracked
        end
    end

    retval_maybe_tracked = exec(gen_fn, state, args_maybe_tracked)

    # restore previous tracked parameter scope
    state.tracked_params = prev_tracked_params

    retval_maybe_tracked
end

@noinline function ReverseDiff.special_reverse_exec!(
        instruction::ReverseDiff.SpecialInstruction{BackpropParamsRecord})
    record = instruction.func
    gen_fn = record.gen_fn
    args_maybe_tracked = instruction.input
    retval_maybe_tracked = instruction.output
    if accepts_output_grad(gen_fn)
        @assert istracked(retval_maybe_tracked)
        retval_grad = deriv(retval_maybe_tracked)
    else
        @assert !istracked(retval_maybe_tracked)
        retval_grad = nothing
    end
    arg_grads = accumulate_param_gradients!(record.subtrace, retval_grad, record.scaler)

    # if has_argument_grads(gen_fn) is true for a given argument, then that
    # argument may or may not be tracked. if has_argument_grads(gen_fn) is
    # false for a given argument, then that argument must not be tracked.
    # otherwise it is an error.
    # note: code duplication with accumulate_param_gradients!
    for (i, (arg, grad, has_grad)) in enumerate(
            zip(args_maybe_tracked, arg_grads, has_argument_grads(gen_fn)))
        if has_grad && istracked(arg)
            increment_deriv!(arg, grad)
        elseif !has_grad && istracked(arg)
            error("Gradient required but not available for argument $i of $gen_fn")
        end
    end
    nothing
end

function accumulate_param_gradients!(trace::DynamicDSLTrace, retval_grad, scaler=1.)
    gen_fn = trace.gen_fn
    tape = new_tape()
    state = GFBackpropParamsState(trace, tape, gen_fn.params, scaler)
    args = get_args(trace)
    args_maybe_tracked = (map(maybe_track,
        args, gen_fn.has_argument_grads, fill(tape, length(args)))...,)
    retval_maybe_tracked = exec(gen_fn, state, args_maybe_tracked)

    # note: code duplication with choice_gradients
    if accepts_output_grad(gen_fn)
        if retval_grad == nothing
            error("Return value gradient required but not provided")
        end
        if istracked(retval_maybe_tracked)
            deriv!(retval_maybe_tracked, retval_grad)
        end
        # note: if accepts_output_grad(gen_fn) and
        # !istracked(retval_maybe_tracked), this means the return value did not
        # depend on the gradient source elements for this trace. that is okay.
    else
        if retval_grad != nothing
            error("Return value gradient not supported, but got $retval_grad != nothing")
        end
    end
    seed!(state.score)
    reverse_pass!(tape)

    # increment the gradient accumulators for trainable parameters in scope
    for (name, tracked) in state.tracked_params
        gen_fn.params_grad[name] += deriv(tracked) * state.scaler
    end

    # increment the gradient accumulators for trainable parameters in splice calls
    for ((spliced_gen_fn, name), tracked) in state.splice_tracked_params
        spliced_gen_fn.params_grad[name] += deriv(tracked) * state.scaler
    end

    # return gradients with respect to arguments with gradients, or nothing
    input_grads::Tuple = (map((arg, has_grad) -> has_grad ? deriv(arg) : nothing,
                             args_maybe_tracked, gen_fn.has_argument_grads)...,)
    input_grads
end


####################
# choice_gradients #
####################

mutable struct GFBackpropTraceState
    trace::DynamicDSLTrace
    score::Float64
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
end

function GFBackpropTraceState(trace, params)
    score = 0.
    visitor = AddressVisitor()
    GFBackpropTraceState(trace, score, visitor, params)
end

# TODO use a custom adjoint for Trie

function get_selected(
        trie::Trie, grad_trie::NamedTuple, selection::Selection, addr_so_far=nothing)
        #get_addr::Function)

    values = choicemap()
    grads = choicemap()

    for (key, record) in get_leaf_nodes(trie) # may be subtrace
        if record.is_choice
            if (addr_so_far == nothing ? key : addr_so_far => key) in selection
                values[key] = record.subtrace_or_retval
                grads[key] = grad_trie.leaf_nodes[key].subtrace_or_retval
            end
        else
            subselection = selection[key]
            if haskey(grad_trie.leaf_nodes, key)
                retval_grad = grad_trie.leaf_nodes[key].subtrace_or_retval
            else
                # TODO every retval needs to have a zero method?
                # (including nothing?)
                retval_grad = zero(get_retval(record.subtrace_or_retval))
            end
            (_, choice_vals, choice_grads) = choice_gradients(
                record.subtrace_or_retval, subselection, retval_grad)
            set_submap!(values, key, choice_vals)
            set_submap!(grads, key, choice_grads)
        end
    end

    for (key, subtrie) in get_internal_nodes(trie)
        grad_subtrie = grad_trie.internal_nodes[key]
        values_submap, grads_submap = get_selected(
                subtrie, grad_subtrie, selection,
                (addr_so_far == nothing ? key : addr_so_far => key))
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

pretend_call(subtrace, args) = get_retval(subtrace)

Zygote.@adjoint pretend_call(subtrace, args) = begin
    retval = pretend_call(subtrace, args)
    fn = (retval_grad) -> begin
        (arg_grads, _, _) = choice_gradients(subtrace, select(), retval_grad)
        # NOTE: we are using the retval_grad as the adjoint for the subtrace
        (retval_grad, arg_grads)
    end
    (retval, fn)
end

function traceat(
        state::GFBackpropTraceState, gen_fn::GenerativeFunction{T,U},
        args, key) where {T,U}
    visit!(state.visitor, key)
    subtrace = get_call(state.trace, key).subtrace
    get_gen_fn(subtrace) === gen_fn || gen_fn_changed_error(key)
    pretend_call(subtrace, args)
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

    fn = (trace) -> begin
        state = GFBackpropTraceState(trace, gen_fn.params)
        retval = exec(gen_fn, state, get_args(trace))
        (state.score, retval)
    end

    _, back = Zygote.pullback(fn, trace)
    trace_grad_ref, = back((1., retval_grad))
    trace_grad = trace_grad_ref[]
    input_grads = trace_grad.args
    grad_trie = trace_grad.trie

    choice_vals, choice_grads = get_selected(
        trace.trie, grad_trie, selection, nothing)

    (input_grads, choice_vals, choice_grads)
end
