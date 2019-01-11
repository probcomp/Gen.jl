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

    # only those tracked parameters that are in scope (not including splice calls)
    tracked_params::Dict{Symbol,Any}

    # tracked parameters for all (nested) @splice calls
    splice_tracked_params::Dict{Tuple{GenerativeFunction,Symbol},Any}
end

function track_params(tape, params)
    tracked_params = Dict{Symbol,Any}()
    for (name, value) in params
        tracked_params[name] = track(value, tape)
    end
    tracked_params
end

function GFBackpropParamsState(trace::DynamicDSLTrace, tape, params)
    tracked_params = track_params(tape, params)
    splice_tracked_params = Dict{Tuple{GenerativeFunction,Symbol},Any}()
    score = track(0., tape)
    GFBackpropParamsState(trace, score, tape, AddressVisitor(), tracked_params,
        splice_tracked_params)
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
    # TODO check has_argument_grads against the tracked vs untracked status of args?
    # TODO use logpdf_grad instead of logpdf?
    state.score += logpdf(dist, retval, args...) 
    retval
end

struct BackpropParamsRecord
    gen_fn::GenerativeFunction
    subtrace::Any
end

function addr(state::GFBackpropParamsState, gen_fn::GenerativeFunction{T},
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
        BackpropParamsRecord(gen_fn, subtrace), (args_maybe_tracked...,), retval_maybe_tracked)
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
            # parameter was already tracked in another @splice
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
    arg_grads = backprop_params(record.subtrace, retval_grad)

    # if has_argument_grads(gen_fn) is true for a given argument, then that
    # argument may or may not be tracked. if has_argument_grads(gen_fn) is
    # false for a given argument, then that argument must not be tracked.
    # otherwise it is an error.
    # note: code duplication with backprop_params
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

function backprop_params(trace::DynamicDSLTrace, retval_grad)
    gen_fn = trace.gen_fn
    tape = new_tape()
    state = GFBackpropParamsState(trace, tape, gen_fn.params)
    args = get_args(trace)
    args_maybe_tracked = (map(maybe_track,
        args, gen_fn.has_argument_grads, fill(tape, length(args)))...,)
    retval_maybe_tracked = exec(gen_fn, state, args_maybe_tracked)

    # note: code duplication with backprop_trace
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

    # increment the gradient accumulators for static parameters in scope
    for (name, tracked) in state.tracked_params
        gen_fn.params_grad[name] += deriv(tracked)
    end

    # increment the gradient accumulators for static parameters in splice calls
    for ((spliced_gen_fn, name), tracked) in state.splice_tracked_params
        spliced_gen_fn.params_grad[name] += deriv(tracked)
    end

    # return gradients with respect to arguments with gradients, or nothing
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
    tracked_choices::Trie{Any,TrackedReal}
    value_assmt::DynamicAssignment
    gradient_assmt::DynamicAssignment
end

function GFBackpropTraceState(trace, selection, params, tape)
    score = track(0., tape)
    visitor = AddressVisitor()
    tracked_choices = Trie{Any,TrackedReal}()
    value_assmt = DynamicAssignment()
    gradient_assmt = DynamicAssignment()
    GFBackpropTraceState(trace, score, tape, visitor, params,
        selection, tracked_choices, value_assmt, gradient_assmt)
end

function fill_gradient_assmt!(gradient_assmt::DynamicAssignment,
                             tracked_trie::Trie{Any,TrackedReal})
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
                          tracked_trie::Trie{Any,TrackedReal})
    for (key, tracked) in get_leaf_nodes(tracked_trie)
        set_value!(value_assmt, key, value(tracked))
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
    # TODO check has_argument_grads against the tracked vs untracked status of args?
    # TODO use logpdf_grad instead of logpdf?
    if has_leaf_node(state.selection, key)
        tracked_retval = track(retval, state.tape)
        set_leaf_node!(state.tracked_choices, key, tracked_retval)
        score_tracked = logpdf(dist, tracked_retval, args...)
        state.score += score_tracked
        return tracked_retval
    else
        state.score += logpdf(dist, retval, args...)
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
              args_maybe_tracked, key) where {T,U}
    local retval::T
    local subtrace::U
    visit!(state.visitor, key)
    if has_leaf_node(state.selection, key)
        error("Cannot select a whole subtrace, tried to select $key")
    end
    subtrace = get_call(state.trace, key).subtrace
    get_gen_fn(subtrace) === gen_fn || gen_fn_changed_error(key)
    retval = get_retval(subtrace)
    if accepts_output_grad(gen_fn)
        retval_maybe_tracked = track(retval, state.tape)
        @assert istracked(retval_maybe_tracked)
    else
        retval_maybe_tracked = retval
        @assert !istracked(retval_maybe_tracked)
    end
    if has_internal_node(state.selection, key)
        selection = get_internal_node(state.selection, key)
    else
        selection = EmptyAddressSet()
    end
    record = BackpropTraceRecord(gen_fn, subtrace, selection, state.value_assmt,
        state.gradient_assmt, key)
    record!(state.tape, ReverseDiff.SpecialInstruction, record, (args_maybe_tracked...,), retval_maybe_tracked)
    retval_maybe_tracked 
end

function splice(state::GFBackpropTraceState, gen_fn::DynamicDSLFunction,
                args_maybe_tracked::Tuple)
    prev_params = state.params
    state.params = gen_fn.params
    retval = exec(gen_fn, state, args_maybe_tracked)
    state.params = prev_params
    retval
end

@noinline function ReverseDiff.special_reverse_exec!(
        instruction::ReverseDiff.SpecialInstruction{BackpropTraceRecord})
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
    (arg_grads, value_assmt, gradient_assmt) = backprop_trace(
        record.subtrace, record.selection, retval_grad)
    @assert isempty(get_subassmt(record.gradient_assmt, record.key))
    @assert !has_value(record.gradient_assmt, record.key)
    set_subassmt!(record.gradient_assmt, record.key, gradient_assmt)
    set_subassmt!(record.value_assmt, record.key, value_assmt)

    # if has_argument_grads(gen_fn) is true for a given argument, then that
    # argument may or may not be tracked. if has_argument_grads(gen_fn) is
    # false for a given argument, then that argument must not be tracked.
    # otherwise it is an error.
    # note: code duplication with backprop_params
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

function backprop_trace(trace::DynamicDSLTrace, selection::AddressSet, retval_grad)
    gen_fn = trace.gen_fn
    tape = new_tape()
    state = GFBackpropTraceState(trace, selection, gen_fn.params, tape)
    args = get_args(trace)
    args_maybe_tracked = (map(maybe_track,
        args, gen_fn.has_argument_grads, fill(tape, length(args)))...,)
    retval_maybe_tracked = exec(gen_fn, state, args_maybe_tracked)

    # note: code duplication with backprop_params
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

    # fill trace gradient with gradients with respect to primitive random choices
    fill_gradient_assmt!(state.gradient_assmt, state.tracked_choices)
    fill_value_assmt!(state.value_assmt, state.tracked_choices)

    # return gradients with respect to arguments with gradients, or nothing
    # NOTE: if a value isn't tracked the gradient is nothing
    input_grads::Tuple = (map((arg, has_grad) -> has_grad ? deriv(arg) : nothing,
                             args_maybe_tracked, gen_fn.has_argument_grads)...,)

    (input_grads, state.value_assmt, state.gradient_assmt)
end
