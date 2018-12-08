import ReverseDiff
using ReverseDiff: SpecialInstruction, istracked, increment_deriv!, deriv, TrackedArray, deriv!
using ReverseDiff: InstructionTape, TrackedReal, seed!, unseed!, reverse_pass!, record!
import ReverseDiff: special_reverse_exec!, track

function maybe_track(arg, has_argument_grad::Bool, tape)
    has_argument_grad ? track(arg, tape) : arg
end

###################
# backprop_params #
###################

mutable struct GFBackpropParamsState
    trace::DynamicDSLTrace
    score::TrackedReal
    tape::InstructionTape
    visitor::AddressVisitor
    tracked_params::Dict{Symbol,Any}
end

function GFBackpropParamsState(trace::DynamicDSLTrace, tape, params)
    tracked_params = Dict{Symbol,Any}()
    for (name, value) in params
        tracked_params[name] = track(value, tape)
    end
    score = track(0., tape)
    GFBackpropParamsState(trace, score, tape, AddressVisitor(), tracked_params)
end

function read_param(state::GFBackpropParamsState, name::Symbol)
    value = state.tracked_params[name]
    value
end

function addr(state::GFBackpropParamsState, dist::Distribution{T},
              args, key) where {T}
    local retval::T
    visit!(state.visitor, key)
    retval = get_choice(state.trace, key).retval
    state.score += logpdf(dist, retval, args...) # TODO use logpdf_grad?
    retval
end

struct BackpropParamsRecord
    gen_fn::GenerativeFunction
    subtrace::Any
end

function addr(state::GFBackpropParamsState, gen_fn::GenerativeFunction{T},
              args, key) where {T}
    local retval::T
    visit!(state.visitor, key)
    subtrace = get_call(state.trace, key).subtrace
    retval = get_retval(subtrace)
    retval_maybe_tracked = track(retval, state.tape)
    # some of the args may be tracked (see special_reverse_exec!)
    # note: we still need to run backprop_params on gen_Fn even if it does not
    # accept an output gradient, because it may make random choices.
    record!(state.tape, SpecialInstruction,
        BackpropParamsRecord(gen_fn, subtrace), (args...,), retval_maybe_tracked)
    retval_maybe_tracked 
end

function splice(state::GFBackpropParamsState, gen_fn::DynamicDSLFunction,
                args::Tuple)
    exec(gen_fn, state, args)
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{BackpropParamsRecord})
    record = instruction.func
    gen = record.gen_fn
    args_maybe_tracked = instruction.input
    retval_maybe_tracked = instruction.output
    if istracked(retval_maybe_tracked)
        retval_grad = deriv(retval_maybe_tracked)
    else
        retval_grad = nothing
    end
    arg_grads = backprop_params(gen, record.subtrace, retval_grad)
    for (arg, grad, has_grad) in zip(args_maybe_tracked, arg_grads, has_argument_grads(gen))
        if has_grad && istracked(arg)
            increment_deriv!(arg, grad)
        end
    end
    nothing
end

function backprop_params(gen_fn::DynamicDSLFunction, trace::DynamicDSLTrace, retval_grad)
    tape = InstructionTape()
    state = GFBackpropParamsState(trace, tape, gen_fn.params)
    call = get_call_record(trace)
    args = call.args
    args_maybe_tracked = (map(maybe_track, args, gen_fn.has_argument_grads, fill(tape, length(args)))...,)
    retval_maybe_tracked = exec(gen_fn, state, args_maybe_tracked)
    if istracked(retval_maybe_tracked)
        deriv!(retval_maybe_tracked, retval_grad)
    end
    seed!(state.score)
    reverse_pass!(tape)

    # increment the gradient accumulators for static parameters
    for (name, tracked) in state.tracked_params
        gen_fn.params_grad[name] += deriv(tracked)
    end

    # return gradients with respect to inputs
    # NOTE: if a value isn't tracked the gradient is nothing
    input_grads::Tuple = (map((arg, has_grad) -> has_grad ? deriv(arg) : nothing,
                             args_maybe_tracked, gen_fn.has_argument_grads)...,)
    input_grads
end


##################
# backprop_trace #
##################

mutable struct GFBackpropTraceState
    trace::DynamicDSLTrace
    score::TrackedReal
    tape::InstructionTape
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
    selection::AddressSet
    tracked_choices::HomogenousTrie{Any,TrackedReal}
    value_assmt::DynamicAssignment
    gradient_assmt::DynamicAssignment
end

function GFBackpropTraceState(trace, selection, params, tape)
    score = track(0., tape)
    visitor = AddressVisitor()
    tracked_choices = HomogenousTrie{Any,TrackedReal}()
    value_assmt = DynamicAssignment()
    gradient_assmt = DynamicAssignment()
    GFBackpropTraceState(trace, score, tape, visitor, params,
        selection, tracked_choices, value_assmt, gradient_assmt)
end

function fill_gradient_assmt!(gradient_assmt::DynamicAssignment,
                             tracked_trie::HomogenousTrie{Any,TrackedReal})
    for (key, tracked) in get_leaf_nodes(tracked_trie)
        set_value!(gradient_assmt, key, deriv(tracked))
    end
    # NOTE: there should be no address collision between these primitive
    # choices and the gen_fn invocations, as enforced by the visitor
    for (key, subtrie) in get_internal_nodes(tracked_trie)
        @assert !has_value(gradient_assmt, key) && isempty(get_subassmt(gradient_assmt, key))
        gradient_subssmt = DynamicAssignment()
        fill_gradient_assmt!(gradient_subassmt, subtrie)
        set_subassmt!(gradient_assmt, key, gradient_subassmt)
    end
end

function fill_value_assmt!(value_assmt::DynamicAssignment,
                          tracked_trie::HomogenousTrie{Any,TrackedReal})
    for (key, tracked) in get_leaf_nodes(tracked_trie)
        set_value!(value_assmt, key, ReverseDiff.value(tracked))
    end
    # NOTE: there should be no address collision between these primitive
    # choices and the gen_fn invocations, as enforced by the visitor
    for (key, subtrie) in get_internal_nodes(tracked_trie)
        @assert !has_value(value_assmt, key) && !isempty(get_subassmt(value_assmt, key))
        value_assmt_subassmt = DynamicAssignment()
        fill_value_assmt!(value_assmt_subassmt, subtrie)
        set_subassmt!(value_assmt, key, value_assmt_subassmt)
    end
end

function addr(state::GFBackpropTraceState, dist::Distribution{T},
              args, key) where {T}
    local retval::T
    visit!(state.visitor, key)
    retval = get_choice(state.trace, key).retval
    if has_internal_node(state.selection, key)
        error("Got internal node but expected leaf node in selection at $key")
    end
    if has_leaf_node(state.selection, key)
        tracked_retval = track(retval, state.tape)
        set_leaf_node!(state.tracked_choices, key, tracked_retval)
        score_tracked = logpdf(dist, tracked_retval, args...) # TODO use logpdf_grad?
        state.score += score_tracked
        return tracked_retval
    else
        state.score += logpdf(dist, retval, args...) # TODO use logpdf_grad?
        return retval
    end
end

struct BackpropTraceRecord
    gen_fn::GenerativeFunction
    subtrace::Any
    selection::AddressSet
    value_assmt::DynamicAssignment
    gradient_assmt::DynamicAssignment
    key::Any
end

function addr(state::GFBackpropTraceState, gen_fn::GenerativeFunction{T,U},
              args, key) where {T,U}
    local retval::T
    local subtrace::U
    visit!(state.visitor, key)
    if has_leaf_node(state.selection, key)
        error("Cannot select a whole subtrace, tried to select $key")
    end
    subtrace = get_call(state.trace, key).subtrace
    retval = get_retval(subtrace)
    retval_maybe_tracked = track(retval, state.tape)

    # some of the args may be tracked (see special_reverse_exec!)
    # note: we still need to run backprop_params on gen_fn, even if it does not
    # accept an output gradient, because it may make random choices.
    if has_internal_node(state.selection, key)
        selection = get_internal_node(state.selection, key)
    else
        selection = EmptyAddressSet()
    end
    record = BackpropTraceRecord(gen_fn, trace, selection, state.value_assmt,
        state.gradient_assmt, key)
    record!(state.tape, SpecialInstruction, record, (args...,), retval_maybe_tracked)
    retval_maybe_tracked 
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{BackpropTraceRecord})
    record = instruction.func
    gen_fn = record.gen_fn
    args_maybe_tracked = instruction.input
    retval_maybe_tracked = instruction.output
    if istracked(retval_maybe_tracked)
        retval_grad = deriv(retval_maybe_tracked)
    else
        retval_grad = nothing
    end
    (arg_grads, value_assmt, gradient_assmt) = backprop_trace(
        gen_fn, record.subtrace, record.selection, retval_grad)

    @assert !isempty(get_subassmt(record.gradient_assmt, record.key))
    @assert !has_value(record.gradient_assmt, record.key)
    set_subassmt!(record.gradient_assmt, record.key, gradient_assmt)
    set_subassmt!(record.value_assmt, record.key, value_assmt)

    for (arg, grad, has_grad) in zip(args_maybe_tracked, arg_grads, has_argument_grads(gen_fn))
        if has_grad && istracked(arg)
            increment_deriv!(arg, grad)
        end
    end
    nothing
end

function backprop_trace(gen_fn::DynamicDSLFunction, trace::DynamicDSLTrace,
                        selection::AddressSet, retval_grad)
    @assert gen_fn === trace.gen_fn
    tape = InstructionTape()
    state = GFBackpropTraceState(trace, selection, gen_fn.params, tape)
    args = get_args(trace)
    args_maybe_tracked = (map(maybe_track, args, gen_fn.has_argument_grads, fill(tape, length(args)))...,)
    retval_maybe_tracked = exec(gen_fn, state, args_maybe_tracked)
    if istracked(retval_maybe_tracked)
        deriv!(retval_maybe_tracked, retval_grad)
    end
    seed!(state.score)
    reverse_pass!(tape)

    # fill trace gradient with gradients with respect to primitive random choices
    fill_gradient_assmt!(state.gradient_assmt, state.tracked_choices)
    fill_value_assmt!(state.value_assmt, state.tracked_choices)

    # return gradients with respect to inputs
    # NOTE: if a value isn't tracked the gradient is nothing
    input_grads::Tuple = (map((arg, has_grad) -> has_grad ? deriv(arg) : nothing,
                             args_maybe_tracked, gen_fn.has_argument_grads)...,)

    (input_grads, state.value_assmt, state.gradient_assmt)
end
