import ReverseDiff
using ReverseDiff: SpecialInstruction, istracked, increment_deriv!, deriv, TrackedArray
using ReverseDiff: track, InstructionTape, TrackedReal, seed!, unseed!, reverse_pass!, record!
import ReverseDiff: special_reverse_exec!

mutable struct GFBackpropTraceState
    trace::GFTrace
    score::TrackedReal
    tape::InstructionTape
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
    selection::AddressSet
    tracked_choices::HomogenousTrie{Any,TrackedReal}
    value_trie::DynamicChoiceTrie
    gradient_trie::DynamicChoiceTrie
end

function GFBackpropTraceState(trace, selection, params, tape)
    score = track(0., tape)
    visitor = AddressVisitor()
    tracked_choices = HomogenousTrie{Any,TrackedReal}()
    value_trie = DynamicChoiceTrie()
    gradient_trie = DynamicChoiceTrie()
    GFBackpropTraceState(trace, score, tape, visitor, params,
                       selection, tracked_choices, value_trie, gradient_trie)
end

get_args_change(state::GFBackpropTraceState) = nothing
get_addr_change(state::GFBackpropTraceState, addr) = nothing
set_ret_change!(state::GFBackpropTraceState, value) = begin end

function fill_gradient_trie!(gradient_trie::DynamicChoiceTrie,
                             tracked_trie::HomogenousTrie{Any,TrackedReal})
    for (key, tracked) in get_leaf_nodes(tracked_trie)
        set_leaf_node!(gradient_trie, key, deriv(tracked))
    end
    # NOTE: there should be no address collision between these primitive
    # choices and the generator invocations, as enforced by the visitor
    for (key, node) in get_internal_nodes(tracked_trie)
        @assert !has_leaf_node(gradient_trie, key) && !has_internal_node(gradient_trie, key)
        gradient_trie_node = DynamicChoiceTrie()
        fill_gradient_trie!(gradient_trie_node, node)
        set_internal_node!(gradient_trie, key, gradient_trie_node)
    end
end

function fill_value_trie!(value_trie::DynamicChoiceTrie,
                          tracked_trie::HomogenousTrie{Any,TrackedReal})
    for (key, tracked) in get_leaf_nodes(tracked_trie)
        set_leaf_node!(value_trie, key, ReverseDiff.value(tracked))
    end
    # NOTE: there should be no address collision between these primitive
    # choices and the generator invocations, as enforced by the visitor
    for (key, node) in get_internal_nodes(tracked_trie)
        @assert !has_leaf_node(value_trie, key) && !has_internal_node(value_trie, key)
        value_trie_node = DynamicChoiceTrie()
        fill_value_trie!(value_trie_node, node)
        set_internal_node!(value_trie, key, value_trie_node)
    end
end

function addr(state::GFBackpropTraceState, dist::Distribution{T}, args, addr) where {T}
    visit!(state.visitor, addr)
    call::CallRecord = get_primitive_call(state.trace, addr)
    retval::T = call.retval
    if has_internal_node(state.selection, addr)
        error("Got internal node but expected leaf node in selection at $addr")
    end
    if has_leaf_node(state.selection, addr)
        tracked_retval = track(retval, state.tape)
        set_leaf_node!(state.tracked_choices, addr, tracked_retval)
        score_tracked = logpdf(dist, tracked_retval, args...)
        state.score += score_tracked
        return tracked_retval
    else
        state.score += logpdf(dist, retval, args...)
        return retval
    end
end

struct BackpropTraceRecord
    generator::Generator
    subtrace::Any
    selection::AddressSet
    value_trie::DynamicChoiceTrie
    gradient_trie::DynamicChoiceTrie
    addr::Any
end

function addr(state::GFBackpropTraceState, gen::Generator{T}, args, addr, args_change) where {T}
    visit!(state.visitor, addr)
    if has_leaf_node(state.selection, addr)
        error("Cannot select a whole subtrace, tried to select $addr")
    end
    trace = get_subtrace(state.trace, addr)
    call::CallRecord = get_call_record(trace) # use the return value recorded in the trace
    retval::T = call.retval
    if accepts_output_grad(gen)
        retval_maybe_tracked = track(retval, state.tape)
        if !istracked(retval_maybe_tracked)
            error("Could not track return value at address $addr on AD tape.")
        end
    else
        retval_maybe_tracked = retval
    end
    # some of the args may be tracked (see special_reverse_exec!)
    # note: we still need to run backprop_params on gen, even if it does not
    # accept an output gradient, because it may make random choices.
    if has_internal_node(state.selection, addr)
        selection = get_internal_node(state.selection, addr)
    else
        selection = EmptyAddressSet()
    end
    record = BackpropTraceRecord(gen, trace, selection, state.value_trie,
                                 state.gradient_trie, addr)
    record!(state.tape, SpecialInstruction, record, (args...,), retval_maybe_tracked)
    retval_maybe_tracked 
end

function backprop_trace(gf::GenFunction, trace::GFTrace, selection::AddressSet, retval_grad)
    tape = InstructionTape()
    state = GFBackpropTraceState(trace, selection, gf.params, tape)
    call = get_call_record(trace)
    args = call.args
    args_maybe_tracked = (map(maybe_track, args, gf.has_argument_grads, fill(tape, length(args)))...,)
    retval_maybe_tracked = exec(gf, state, args_maybe_tracked)
    if gf.accepts_output_grad
        if !istracked(retval_maybe_tracked)
            error("Output of $gf accepts gradient but return value was not tracked on AD tape.")
        end
        if retval_grad != nothing
            # it is not an error if retval_grad is nothing, it means the gradient is zero
            # this is a convenience that 
            deriv!(retval_maybe_tracked, retval_grad)
        end
    end
    seed!(state.score)
    reverse_pass!(tape)

    # fill trace gradient with gradients with respect to primitive random choices
    fill_gradient_trie!(state.gradient_trie, state.tracked_choices)
    fill_value_trie!(state.value_trie, state.tracked_choices)

    # return gradients with respect to inputs
    # NOTE: if a value isn't tracked the gradient is nothing
    input_grads::Tuple = (map((arg, has_grad) -> has_grad ? deriv(arg) : nothing,
                             args_maybe_tracked, gf.has_argument_grads)...,)

    (input_grads, state.value_trie, state.gradient_trie)
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{BackpropTraceRecord})
    record = instruction.func
    gen = record.generator
    args_maybe_tracked = instruction.input
    retval_maybe_tracked = instruction.output
    if accepts_output_grad(gen)
        @assert istracked(retval_maybe_tracked)
        retval_grad = deriv(retval_maybe_tracked)
    else
        retval_grad = nothing
    end
    (arg_grads, value_trie, grad_trie) = backprop_trace(
        gen, record.subtrace, record.selection, retval_grad)

    @assert !has_internal_node(record.gradient_trie, record.addr)
    @assert !has_leaf_node(record.gradient_trie, record.addr)
    set_internal_node!(record.gradient_trie, record.addr, grad_trie)
    set_internal_node!(record.value_trie, record.addr, value_trie)

    for (arg, grad, has_grad) in zip(args_maybe_tracked, arg_grads, has_argument_grads(gen))
        if has_grad && istracked(arg)
            increment_deriv!(arg, grad)
        end
    end
    nothing
end
