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
    scale_factor::Float64

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

function GFBackpropParamsState(trace::DynamicDSLTrace, tape, params, scale_factor)
    tracked_params = track_params(tape, params)
    splice_tracked_params = Dict{Tuple{GenerativeFunction,Symbol},Any}()
    score = track(0., tape)
    GFBackpropParamsState(trace, score, tape, AddressVisitor(), scale_factor,
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
    scale_factor::Float64
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
        BackpropParamsRecord(gen_fn, subtrace, state.scale_factor),
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
    arg_grads = accumulate_param_gradients!(record.subtrace, retval_grad, record.scale_factor)

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

function set_retval_grad!(gen_fn::DynamicDSLFunction, retval_maybe_tracked, retval_grad)
    if accepts_output_grad(gen_fn)
        if istracked(retval_maybe_tracked)
            deriv!(retval_maybe_tracked, retval_grad)
        end
    else
        error("Return value gradient not supported, but got gradient $retval_grad != nothing")
    end
end

function set_retval_grad!(gen_fn::DynamicDSLFunction, retval_maybe_tracked, retval_grad::Nothing)
    # don't accumulate the gradient for retval_maybe_tracked, if it is tracked
end


function accumulate_param_gradients!(trace::DynamicDSLTrace, retval_grad, scale_factor=1.)
    gen_fn = trace.gen_fn
    tape = new_tape()
    state = GFBackpropParamsState(trace, tape, gen_fn.params, scale_factor)
    args = get_args(trace)
    args_maybe_tracked = (map(maybe_track,
        args, gen_fn.has_argument_grads, fill(tape, length(args)))...,)
    retval_maybe_tracked = exec(gen_fn, state, args_maybe_tracked)
    set_retval_grad!(gen_fn, retval_maybe_tracked, retval_grad)
    seed!(state.score)
    reverse_pass!(tape)

    # increment the gradient accumulators for trainable parameters in scope
    for (name, tracked) in state.tracked_params
        gen_fn.params_grad[name] += deriv(tracked) * state.scale_factor
    end

    # increment the gradient accumulators for trainable parameters in splice calls
    for ((spliced_gen_fn, name), tracked) in state.splice_tracked_params
        spliced_gen_fn.params_grad[name] += deriv(tracked) * state.scale_factor
    end

    # return gradients with respect to arguments with gradients, or nothing
    input_grads::Tuple = (map((arg, has_grad) -> has_grad ? deriv(arg) : nothing,
                             args_maybe_tracked, gen_fn.has_argument_grads)...,)
    input_grads
end


##################
# choice_gradients #
##################

mutable struct GFBackpropTraceState
    trace::DynamicDSLTrace
    score::TrackedReal
    tape::InstructionTape
    visitor::AddressVisitor
    params::Dict{Symbol,Any}
    selection::Selection
    tracked_choices::Trie{Any,Union{TrackedReal,TrackedArray}}
    value_choices::DynamicChoiceMap
    gradient_choices::DynamicChoiceMap
end

function GFBackpropTraceState(trace, selection, params, tape)
    score = track(0., tape)
    visitor = AddressVisitor()
    tracked_choices = Trie{Any,Union{TrackedReal,TrackedArray}}()
    value_choices = choicemap()
    gradient_choices = choicemap()
    GFBackpropTraceState(trace, score, tape, visitor, params,
        selection, tracked_choices, value_choices, gradient_choices)
end

function fill_submaps!(
        map::DynamicChoiceMap,
        tracked_trie::Trie{Any,Union{TrackedReal,TrackedArray}},
        mode)
    # NOTE: there should be no address collision between these primitive
    # choices and the gen_fn invocations, as enforced by the visitor
    for (key, subtrie) in get_internal_nodes(tracked_trie)
        @assert !has_value(map, key) && isempty(get_submap(map, key))
        submap= choicemap()
        fill_map!(submap, subtrie, mode)
        set_submap!(map, key, submap)
    end
end

function fill_map!(
        map::DynamicChoiceMap,
        tracked_trie::Trie{Any,Union{TrackedReal,TrackedArray}},
        mode::Val{:gradient_map})
    for (key, tracked) in get_leaf_nodes(tracked_trie)
        set_value!(map, key, deriv(tracked))
    end
    fill_submaps!(map, tracked_trie, mode)
end

function fill_map!(
        map::DynamicChoiceMap,
        tracked_trie::Trie{Any,Union{TrackedReal,TrackedArray}},
        mode::Val{:value_map})
    for (key, tracked) in get_leaf_nodes(tracked_trie)
        set_value!(map, key, value(tracked))
    end
    fill_submaps!(map, tracked_trie, mode)
end

function traceat(state::GFBackpropTraceState, dist::Distribution{T},
              args_maybe_tracked, key) where {T}
    local retval::T
    visit!(state.visitor, key)
    retval = get_choice(state.trace, key).retval
    args = map(value, args_maybe_tracked)
    score_tracked = track(logpdf(dist, retval, args...), state.tape)
    if key in state.selection
        tracked_retval = track(retval, state.tape)
        set_leaf_node!(state.tracked_choices, key, tracked_retval)
        record!(state.tape, ReverseDiff.SpecialInstruction, dist,
            (tracked_retval, args_maybe_tracked...,), score_tracked)
        state.score += score_tracked
        return tracked_retval
    else
        record!(state.tape, ReverseDiff.SpecialInstruction, dist,
            (retval, args_maybe_tracked...,), score_tracked)
        state.score += score_tracked
        return retval
    end
end

struct BackpropTraceRecord
    gen_fn::GenerativeFunction
    subtrace::Any
    selection::Selection
    value_choices::DynamicChoiceMap
    gradient_choices::DynamicChoiceMap
    key::Any
end

function traceat(state::GFBackpropTraceState, gen_fn::GenerativeFunction{T,U},
              args_maybe_tracked, key) where {T,U}
    local retval::T
    local subtrace::U
    visit!(state.visitor, key)
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
    selection = state.selection[key]
    record = BackpropTraceRecord(gen_fn, subtrace, selection, state.value_choices,
        state.gradient_choices, key)
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
    (arg_grads, value_choices, gradient_choices) = choice_gradients(
        record.subtrace, record.selection, retval_grad)
    @assert isempty(get_submap(record.gradient_choices, record.key))
    @assert !has_value(record.gradient_choices, record.key)
    set_submap!(record.gradient_choices, record.key, gradient_choices)
    set_submap!(record.value_choices, record.key, value_choices)

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

function choice_gradients(trace::DynamicDSLTrace, selection::Selection, retval_grad)
    gen_fn = trace.gen_fn
    tape = new_tape()
    state = GFBackpropTraceState(trace, selection, gen_fn.params, tape)
    args = get_args(trace)
    args_maybe_tracked = (map(maybe_track,
        args, gen_fn.has_argument_grads, fill(tape, length(args)))...,)
    retval_maybe_tracked = exec(gen_fn, state, args_maybe_tracked)
    set_retval_grad!(gen_fn, retval_maybe_tracked, retval_grad)
    seed!(state.score)
    reverse_pass!(tape)

    # fill trace gradient with gradients with respect to primitive random choices
    fill_map!(state.gradient_choices, state.tracked_choices, Val{:gradient_map}())
    fill_map!(state.value_choices, state.tracked_choices, Val{:value_map}())

    # return gradients with respect to arguments with gradients, or nothing
    # NOTE: if a value isn't tracked the gradient is nothing
    input_grads::Tuple = (map((arg, has_grad) -> has_grad ? deriv(arg) : nothing,
                             args_maybe_tracked, gen_fn.has_argument_grads)...,)

    (input_grads, state.value_choices, state.gradient_choices)
end
