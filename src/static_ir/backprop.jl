# TODO this code needs to be simplified

struct BackpropTraceMode end
struct BackpropParamsMode end

const gradient_prefix = gensym("gradient")
gradient_var(node::StaticIRNode) = Symbol("$(gradient_prefix)_$(node.name)")

const value_trie_prefix = gensym("value_trie")
value_trie_var(node::GenerativeFunctionCallNode) = Symbol("$(value_trie_prefix)_$(node.addr)")

const gradient_trie_prefix = gensym("gradient_trie")
gradient_trie_var(node::GenerativeFunctionCallNode) = Symbol("$(gradient_trie_prefix)_$(node.addr)")

const tape_prefix = gensym("tape")
tape_var(node::JuliaNode) = Symbol("$(tape_prefix)_$(node.name)")

const maybe_tracked_value_prefix = gensym("maybe_tracked_value")
maybe_tracked_value_var(node::JuliaNode) = Symbol("$(maybe_tracked_value_prefix)_$(node.name)")

const maybe_tracked_arg_prefix = gensym("maybe_tracked_arg")
maybe_tracked_arg_var(node::JuliaNode, i::Int) = Symbol("$(maybe_tracked_arg_prefix)_$(node.name)_$i")

function forward_marking!(
        selected_choices, selected_calls, fwd_marked,
        node::TrainableParameterNode, mode)
    if mode == BackpropParamsMode()
        push!(fwd_marked, node)
    end
end

function forward_marking!(selected_choices, selected_calls, fwd_marked, node::ArgumentNode, mode)
    if node.compute_grad
        push!(fwd_marked, node)
    end
end

function forward_marking!(selected_choices, selected_calls, fwd_marked, node::JuliaNode, mode)
    if any(input_node in fwd_marked for input_node in node.inputs)
        push!(fwd_marked, node)
    end
end

function forward_marking!(selected_choices, selected_calls, fwd_marked, node::RandomChoiceNode, mode)
    if node in selected_choices
        push!(fwd_marked, node)
    end
end

function forward_marking!(selected_choices, selected_calls, fwd_marked, node::GenerativeFunctionCallNode, mode)
    if node in selected_calls || any(input_node in fwd_marked for input_node in node.inputs)
        push!(fwd_marked, node)
    end
end

function backward_marking!(back_marked, node::TrainableParameterNode) end

function backward_marking!(back_marked, node::ArgumentNode) end

function backward_marking!(back_marked, node::JuliaNode)
    if node in back_marked
        for input_node in node.inputs
            push!(back_marked, input_node)
        end
    end
end

function backward_marking!(back_marked, node::RandomChoiceNode)
    # the logpdf of every random choice is a SINK
    for input_node in node.inputs
        push!(back_marked, input_node)
    end
    # the value of every random choice is in back_marked, since it affects its logpdf
    push!(back_marked, node)
end

function backward_marking!(back_marked, node::GenerativeFunctionCallNode)
    # the logpdf of every generative function call is a SINK
    # (we could ask whether the generative function is deterministic or not
    # as a perforance optimization, because only stochsatic generative functions 
    # actually have a non-trivial logpdf)
    for (input_node, has_grad) in zip(node.inputs, has_argument_grads(node.generative_function))
        if has_grad
            push!(back_marked, input_node)
        end
    end
end

function forward_codegen!(stmts, fwd_marked, back_marked, node::TrainableParameterNode)
    if node in back_marked
        push!(stmts, :($(node.name) = $(QuoteNode(get_parameter_value))(trace, $(QuoteNode(node.name)))))
    end

    if node in fwd_marked && node in back_marked

        # initialize gradient to zero
        # NOTE: we are avoiding allocating a new gradient accumulator for this function
        # instead, we are using the threadsafe gradient accumulator directly..
        push!(stmts, :($(gradient_var(node)) = $(QuoteNode(get_gradient_accumulator))(trace, $(QuoteNode(node.name)))))
    end
end

function forward_codegen!(stmts, fwd_marked, back_marked, node::ArgumentNode)
    if node in fwd_marked && node in back_marked

        # initialize gradient to zero
        push!(stmts, :($(gradient_var(node)) = zero($(node.name))))
    end
end

function forward_codegen!(stmts, fwd_marked, back_marked, node::JuliaNode)

    if (node in fwd_marked) && (node in back_marked)

        # tracked forward execution
        tape = tape_var(node)
        push!(stmts, :($tape = $(QuoteNode(new_tape))()))
        args_maybe_tracked = Symbol[]
        for (i, input_node) in enumerate(node.inputs)
            arg = input_node.name
            arg_maybe_tracked = maybe_tracked_arg_var(node, i)
            if input_node in fwd_marked
                push!(stmts, :($arg_maybe_tracked = $(QuoteNode(track))($arg, $tape)))
            else
                push!(stmts, :($arg_maybe_tracked = $arg))
            end
            push!(args_maybe_tracked, arg_maybe_tracked)
        end
        maybe_tracked_value = maybe_tracked_value_var(node)
        push!(stmts, :($maybe_tracked_value = $(QuoteNode(node.fn))($(args_maybe_tracked...))))
        push!(stmts, :($(node.name) = $(QuoteNode(value))($maybe_tracked_value)))

        # initialize gradient to zero
        push!(stmts, :($(gradient_var(node)) = zero($(node.name))))

    elseif node in back_marked

        # regular forward execution.
        args = map((input_node) -> input_node.name, node.inputs)
        push!(stmts, :($(node.name) = $(QuoteNode(node.fn))($(args...))))
    end
end

function forward_codegen!(stmts, fwd_marked, back_marked, node::RandomChoiceNode)
    # every random choice is in back_marked, since it affects it logpdf, but
    # also possibly due to other downstream usage of the value
    @assert node in back_marked
    push!(stmts, :($(node.name) = trace.$(get_value_fieldname(node))))

    if node in fwd_marked
        # the only way we are fwd_marked is if this choice was selected

        # initialize gradient with respect to the value of the random choice to zero
        # it will be a runtime error, thrown here, if there is no zero() method
        push!(stmts, :($(gradient_var(node)) = zero($(node.name))))
    end
end

function forward_codegen!(stmts, fwd_marked, back_marked, node::GenerativeFunctionCallNode)

    if node in back_marked
        # for reference by other nodes during backward_codegen!
        subtrace_fieldname = get_subtrace_fieldname(node)
        push!(stmts, :($(node.name) = $(QuoteNode(get_retval))(trace.$subtrace_fieldname)))
    end

    # NOTE: we will still potentially run choice_gradients recursively on the generative function,
    # we just might not use its return value gradient.
    if (node in fwd_marked) && (node in back_marked)
        # we are fwd_marked if an input was fwd_marked, or if we were selected internally
        push!(stmts, :($(gradient_var(node)) = zero($(node.name))))
    end
end

function backward_codegen!(stmts, ir, selected_calls, fwd_marked, back_marked, node::TrainableParameterNode, mode)
    if mode == BackpropParamsMode()

        # handle case when it is the return node
        if node === ir.return_node && node in fwd_marked
            @assert node in back_marked
            push!(stmts, :(isnothing(retval_grad) && error("Required return value gradient but got nothing")))
            push!(stmts, :($(gradient_var(node)) = $(QuoteNode(in_place_add!))(
                     $(gradient_var(node)), retval_grad, scale_factor)))
        end
    end
end

function backward_codegen!(stmts, ir, selected_calls, fwd_marked, back_marked, node::ArgumentNode, mode)

    # handle case when it is the return node
    if node === ir.return_node && node in fwd_marked
        @assert node in back_marked
        push!(stmts, :($(gradient_var(node)) = $(QuoteNode(in_place_add!))(
                $(gradient_var(node)), retval_grad)))
    end
end

function backward_codegen!(stmts, ir, selected_calls, fwd_marked, back_marked, node::JuliaNode, mode)
    # handle case when it is the return node
    if node === ir.return_node && node in fwd_marked
        @assert node in back_marked
        push!(stmts, :($(gradient_var(node)) = $(QuoteNode(in_place_add!))(
                $(gradient_var(node)), retval_grad)))
    end
    if node in back_marked && any(input_node in fwd_marked for input_node in node.inputs)

        # do backward pass through the Julia code
        push!(stmts, :($(QuoteNode(deriv!))($(maybe_tracked_value_var(node)), $(gradient_var(node)))))
        push!(stmts, :($(QuoteNode(reverse_pass!))($(tape_var(node)))))

        # increment gradients of input nodes that are in fwd_marked
        for (i, input_node) in enumerate(node.inputs)
            if input_node in fwd_marked
                arg_maybe_tracked = maybe_tracked_arg_var(node, i)
                if isa(input_node, TrainableParameterNode)
                    @assert mode == BackpropParamsMode()
                    push!(stmts, :($(gradient_var(input_node)) = $(QuoteNode(in_place_add!))(
                        $(gradient_var(input_node)), $(QuoteNode(deriv))($arg_maybe_tracked), scale_factor)))
                else
                    push!(stmts, :($(gradient_var(input_node)) = $(QuoteNode(in_place_add!))(
                        $(gradient_var(input_node)), $(QuoteNode(deriv))($arg_maybe_tracked))))
                end
            end
        end
    end

end

function backward_codegen_random_choice_to_inputs!(
        stmts, ir, fwd_marked, back_marked,
        node::RandomChoiceNode, logpdf_grad::Symbol,
        mode)

    # only evaluate the gradient of the logpdf if we need to
    if any(input_node in fwd_marked for input_node in node.inputs) || node in fwd_marked
        args = map((input_node) -> input_node.name, node.inputs)
        push!(stmts, :($logpdf_grad = $(QuoteNode(Gen.logpdf_grad))($(node.dist), $(node.name), $(args...))))
    end

    # increment gradients of input nodes that are in fwd_marked
    for (i, input_node) in enumerate(node.inputs)
        if input_node in fwd_marked
            @assert input_node in back_marked # this ensured its gradient will have been initialized
            if !has_argument_grads(node.dist)[i]
                error("Distribution $(node.dist) does not have logpdf gradient for argument $i")
            end
            input_node_grad = gradient_var(input_node)
            increment = :($logpdf_grad[$(QuoteNode(i+1))])
            if isa(input_node, TrainableParameterNode) && mode == BackpropParamsMode()
                push!(stmts, :($input_node_grad = $(QuoteNode(in_place_add!))(
                    $input_node_grad, $increment, scale_factor)))
            else
                push!(stmts, :($input_node_grad = $(QuoteNode(in_place_add!))(
                    $input_node_grad, $increment)))
            end
        end
    end

    # handle case when it is the return node
    if node === ir.return_node && node in fwd_marked
        @assert node in back_marked
        push!(stmts, :($(gradient_var(node)) = $(QuoteNode(in_place_add!))(
            $(gradient_var(node)), retval_grad)))
    end
end

function backward_codegen!(stmts, ir, selected_calls, fwd_marked, back_marked,
                       node::RandomChoiceNode, mode::BackpropTraceMode)
    logpdf_grad = gensym("logpdf_grad")

    # backpropagate to the inputs
    backward_codegen_random_choice_to_inputs!(stmts, ir, fwd_marked, back_marked, node, logpdf_grad, mode)

    # backpropagate to the value (if it was selected)
    if node in fwd_marked
        if !has_output_grad(node.dist)
            error("Distribution $dist does not logpdf gradient for its output value")
        end
        push!(stmts, :($(gradient_var(node)) = $(QuoteNode(in_place_add!))(
            $(gradient_var(node)), $logpdf_grad[1])))
    end
end

function backward_codegen!(stmts, ir, selected_calls, fwd_marked, back_marked,
                       node::RandomChoiceNode, mode::BackpropParamsMode)
    logpdf_grad = gensym("logpdf_grad")
    backward_codegen_random_choice_to_inputs!(stmts, ir, fwd_marked, back_marked, node, logpdf_grad, mode)
end

function backward_codegen!(stmts, ir, selected_calls, fwd_marked, back_marked,
                       node::GenerativeFunctionCallNode, mode::BackpropTraceMode)

    # handle case when it is the return node
    if node === ir.return_node && node in fwd_marked
        @assert node in back_marked
        push!(stmts, :($(gradient_var(node)) = $(QuoteNode(in_place_add!))(
                $(gradient_var(node)), retval_grad)))
    end

    if node in fwd_marked
        input_grads = gensym("call_input_grads")
        value_trie = value_trie_var(node)
        gradient_trie = gradient_trie_var(node)
        subtrace_fieldname = get_subtrace_fieldname(node)
        call_selection = gensym("call_selection")
        if node in selected_calls
            push!(stmts, :($call_selection = $(GlobalRef(Gen, :static_getindex))(selection, $(QuoteNode(Val(node.addr))))))
        else
            push!(stmts, :($call_selection = EmptySelection()))
        end
        retval_grad = node in back_marked ? gradient_var(node) : :(nothing)
        push!(stmts, :(($input_grads, $value_trie, $gradient_trie) = $(QuoteNode(choice_gradients))(
            trace.$subtrace_fieldname, $call_selection, $retval_grad)))
    end

    # increment gradients of input nodes that are in fwd_marked
    for (i, input_node) in enumerate(node.inputs)
        if input_node in fwd_marked
            @assert input_node in back_marked # this ensured its gradient will have been initialized
            input_node_grad = gradient_var(input_node)
            increment = :($input_grads[$(QuoteNode(i))])
            push!(stmts, :($(gradient_var(input_node)) = $(QuoteNode(in_place_add!))(
                $input_node_grad, $increment)))
        end
    end

    # NOTE: the value_trie and gradient_trie are dealt with later
end

function backward_codegen!(stmts, ir, selected_calls, fwd_marked, back_marked,
                       node::GenerativeFunctionCallNode, mode::BackpropParamsMode)

    # handle case when it is the return node
    if node === ir.return_node && node in fwd_marked
        @assert node in back_marked
        push!(stmts, :($(gradient_var(node)) = $(QuoteNode(in_place_add!))(
                $(gradient_var(node)), retval_grad)))
    end

    if node in fwd_marked
        input_grads = gensym("call_input_grads")
        subtrace_fieldname = get_subtrace_fieldname(node)
        retval_grad = node in back_marked ? gradient_var(node) : :(nothing)
        push!(stmts, :($input_grads = $(QuoteNode(accumulate_param_gradients!))(trace.$subtrace_fieldname, $retval_grad, scale_factor)))
    end

    # increment gradients of input nodes that are in fwd_marked
    for (i, (input_node, has_grad)) in enumerate(zip(node.inputs, has_argument_grads(node.generative_function)))
        if input_node in fwd_marked && has_grad
            @assert input_node in back_marked # this ensured its gradient will have been initialized
            input_node_grad = gradient_var(input_node)
            increment = :($input_grads[$(QuoteNode(i))])
            if isa(input_node, TrainableParameterNode)
                push!(stmts, :($(gradient_var(input_node)) = $(QuoteNode(in_place_add!))(
                    $input_node_grad, $increment, scale_factor)))
            else
                push!(stmts, :($(gradient_var(input_node)) = $(QuoteNode(in_place_add!))(
                    $input_node_grad, $increment)))
            end
        end
    end
end

function generate_value_gradient_trie(selected_choices::Set{RandomChoiceNode},
                                      selected_calls::Set{GenerativeFunctionCallNode},
                                      value_trie::Symbol, gradient_trie::Symbol)
    selected_choices_vec = collect(selected_choices)
    quoted_leaf_keys = map((node) -> QuoteNode(node.addr), selected_choices_vec)
    leaf_values = map((node) -> :(trace.$(get_value_fieldname(node))), selected_choices_vec)
    leaf_gradients = map((node) -> gradient_var(node), selected_choices_vec)

    selected_calls_vec = collect(selected_calls)
    quoted_internal_keys = map((node) -> QuoteNode(node.addr), selected_calls_vec)
    internal_values = map((node) -> :(get_choices(trace.$(get_subtrace_fieldname(node)))),
                          selected_calls_vec)
    internal_gradients = map((node) -> gradient_trie_var(node), selected_calls_vec)
    quote
        $value_trie = StaticChoiceMap(
            NamedTuple{($(quoted_leaf_keys...),)}(($(leaf_values...),)),
            NamedTuple{($(quoted_internal_keys...),)}(($(internal_values...),)))
        $gradient_trie = StaticChoiceMap(
            NamedTuple{($(quoted_leaf_keys...),)}(($(leaf_gradients...),)),
            NamedTuple{($(quoted_internal_keys...),)}(($(internal_gradients...),)))
    end
end

function get_selected_choices(::EmptyAddressSchema, ::StaticIR)
    Set{RandomChoiceNode}()
end

function get_selected_choices(::AllAddressSchema, ir::StaticIR)
    Set{RandomChoiceNode}(ir.choice_nodes)
end

function get_selected_choices(schema::StaticAddressSchema, ir::StaticIR)
    selected_choice_addrs = Set(keys(schema))
    selected_choices = Set{RandomChoiceNode}()
    for node in ir.choice_nodes
        if node.addr in selected_choice_addrs
            push!(selected_choices, node)
        end
    end
    return selected_choices
end

function get_selected_calls(::EmptyAddressSchema, ::StaticIR)
    return Set{GenerativeFunctionCallNode}()
end

function get_selected_calls(::AllAddressSchema, ir::StaticIR)
    return Set{GenerativeFunctionCallNode}(ir.call_nodes)
end

function get_selected_calls(schema::StaticAddressSchema, ir::StaticIR)
    selected_call_addrs = Set(keys(schema))
    selected_calls = Set{GenerativeFunctionCallNode}()
    for node in ir.call_nodes
        if node.addr in selected_call_addrs
            push!(selected_calls, node)
        end
    end
    return selected_calls
end

function codegen_choice_gradients(trace_type::Type{T}, selection_type::Type,
                                retval_grad_type::Type) where {T<:StaticIRTrace}
    gen_fn_type = get_gen_fn_type(trace_type)
    schema = get_address_schema(selection_type)

    # convert a hierarchical selection to a static selection if it is not already one
    if !(isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema) || isa(schema, AllAddressSchema))
        return quote choice_gradients(trace, StaticSelection(selection), retval_grad) end
    end

    ir = get_ir(gen_fn_type)
    selected_choices = get_selected_choices(schema, ir)
    selected_calls = get_selected_calls(schema, ir)

    # forward marking pass
    fwd_marked = Set{StaticIRNode}()
    for node in ir.nodes
        forward_marking!(selected_choices, selected_calls, fwd_marked, node, BackpropTraceMode())
    end

    # backward marking pass
    back_marked = Set{StaticIRNode}()
    push!(back_marked, ir.return_node)
    for node in reverse(ir.nodes)
        backward_marking!(back_marked, node)
    end

    stmts = Expr[]

    # unpack arguments from the trace
    arg_names = Symbol[arg_node.name for arg_node in ir.arg_nodes]
    push!(stmts, :($(Expr(:tuple, arg_names...)) = $(QuoteNode(get_args))(trace)))

    # forward code-generation pass (initialize gradients to zero, create needed references)
    for node in ir.nodes
        forward_codegen!(stmts, fwd_marked, back_marked, node)
    end

    # backward code-generation pass (increment gradients)
    for node in reverse(ir.nodes)
        backward_codegen!(stmts, ir, selected_calls, fwd_marked, back_marked, node, BackpropTraceMode())
    end

    # assemble value_trie and gradient_trie
    value_trie = gensym("value_trie")
    gradient_trie = gensym("gradient_trie")
    push!(stmts, generate_value_gradient_trie(selected_choices, selected_calls,
        value_trie, gradient_trie))

    # gradients with respect to inputs
    arg_grad = (node::ArgumentNode) -> node.compute_grad ? gradient_var(node) : :(nothing)
    input_grads = Expr(:tuple, map(arg_grad, ir.arg_nodes)...)

    # return values
    push!(stmts, :(return ($input_grads, $value_trie, $gradient_trie)))

    return Expr(:block, stmts...)
end

function codegen_accumulate_param_gradients!(trace_type::Type{T},
                                 retval_grad_type::Type, scale_factor_type) where {T<:StaticIRTrace}
    gen_fn_type = get_gen_fn_type(trace_type)
    ir = get_ir(gen_fn_type)

    # unlike choice_gradients we don't take gradients w.r.t. the value of random choices
    selected_choices = Set{RandomChoiceNode}()

    # we need to guarantee that we visit every generative function call,
    # because we need to backpropagate to its trainable parameters
    selected_calls = Set{GenerativeFunctionCallNode}(
        node for node in ir.nodes if isa(node, GenerativeFunctionCallNode))

    # forward marking pass (propagate forward from 'sources')
    fwd_marked = Set{StaticIRNode}()
    for node in ir.nodes
        forward_marking!(selected_choices, selected_calls, fwd_marked, node, BackpropParamsMode())
    end

    # backward marking pass (propagate backwards from 'sinks')
    back_marked = Set{StaticIRNode}()
    push!(back_marked, ir.return_node)
    for node in reverse(ir.nodes)
        backward_marking!(back_marked, node)
    end

    stmts = Expr[]

    # unpack arguments from the trace
    arg_names = Symbol[arg_node.name for arg_node in ir.arg_nodes]
    push!(stmts, :($(Expr(:tuple, arg_names...)) = $(QuoteNode(get_args))(trace)))

    # forward code-generation pass
    # any node that is backward-marked creates a variable for its current value
    # any node that is forward-marked and backwards marked initializes a gradient variable
    for node in ir.nodes
        forward_codegen!(stmts, fwd_marked, back_marked, node)
    end

    # backward code-generation pass
    # any node that is forward-marked and backwards marked increments its gradient variable
    for node in reverse(ir.nodes)
        backward_codegen!(stmts, ir, selected_calls, fwd_marked, back_marked, node, BackpropParamsMode())
    end

    # gradients with respect to inputs
    arg_grad = (node::ArgumentNode) -> node.compute_grad ? gradient_var(node) : :(nothing)
    input_grads = Expr(:tuple, map(arg_grad, ir.arg_nodes)...)

    # return values
    push!(stmts, :(return $input_grads))

    return Expr(:block, stmts...)
end


push!(generated_functions, quote
@generated function $(GlobalRef(Gen, :choice_gradients))(
        trace::T, selection::$(QuoteNode(Selection)),
        retval_grad) where {T<:$(QuoteNode(StaticIRTrace))}
    return $(QuoteNode(codegen_choice_gradients))(trace, selection, retval_grad)
end
end)

push!(generated_functions, quote
@generated function $(GlobalRef(Gen, :accumulate_param_gradients!))(
        trace::T, retval_grad, scale_factor) where {T<:$(QuoteNode(StaticIRTrace))}
    return $(QuoteNode(codegen_accumulate_param_gradients!))(trace, retval_grad, scale_factor)
end
end)
