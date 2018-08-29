struct BBBackpropTraceState
    ir::BasicBlockIR
    trace::Symbol
    stmts::Vector{Expr}
    schema::Union{StaticAddressSchema,EmptyAddressSchema}
    value_refs::Dict{ValueNode,Expr}
    grad_vars::Dict{ValueNode,Symbol}

    # maps from address to variable storing floating point value (for leaf) or
    # to choice trie (for internal node)
    values_leaf_nodes::Dict{Symbol,Symbol}
    values_internal_nodes::Dict{Symbol,Symbol}
    gradients_leaf_nodes::Dict{Symbol,Symbol}
    gradients_internal_nodes::Dict{Symbol,Symbol}
end

function BBBackpropTraceState(trace, stmts, schema, value_refs, grad_vars)
    values_leaf_nodes = Dict{Symbol,Symbol}()
    values_internal_nodes = Dict{Symbol,Symbol}()
    gradients_leaf_nodes = Dict{Symbol,Symbol}()
    gradients_internal_nodes = Dict{Symbol,Symbol}()
    BBBackpropTraceState(trace, stmts, schema, value_refs, grad_vars,
        values_leaf_nodes, values_internal_nodes,
        gradients_leaf_nodes, gradients_internal_nodes)
end

function backprop_trace_process!(state::BBBackpropTraceState, node::JuliaNode)
    input_value_refs = [state.value_refs[in_node] for in_node in node.input_nodes]
    input_grad_vars = [state.grad_vars[in_node] for in_node in node.input_nodes]
    output_value_ref = state.value_refs[node.output]
    output_grad_var = state.grad_vars[node.output]
    grad_fn = get_grad_fn(node) # TODO implement, generated at macro-expansion time
    # TODO there may not be gradients in some cases..
    input_grad_increments = [gensym("incr") for _ in node.input_nodes]
    push!(stmts, quote
        ($(input_grad_increments...),) = $grad_fn($(output_value_ref, input_value_refs...))
    end)
    for (grad_var, incr) in zip(input_grad_vars, input_grad_increments)
        push!(stmts, quote
            $grad_var += $incr
        end)
    end
end

function backprop_trace_process!(state::BBBackpropTraceState, node::ArgsChangeNode)
    # skip
end

function backprop_trace_process!(state::BBBackpropTraceState, node::AddrChangeNode)
    # skip
end

# TODO should there be a 'has_input_grads' and 'has_output_grad' for each distribution too?
# (like generators)
# and also for Julia expressions?

function backprop_trace_process!(state::BBBackpropTraceState, node::AddrDistNode)
    input_value_refs = [state.value_refs[in_node] for in_node in node.input_nodes]
    input_grad_vars = [state.grad_vars[in_node] for in_node in node.input_nodes]
    output_value_ref = state.value_refs[node.output]
    output_grad_var = state.grad_vars[node.output]
    addr = node.address
    dist = QuoteNode(node.dist)
    output_grad_incr = gensym("incr")
    input_grad_incrs = [gensym("incr") for _ in node.input_nodes]
    # TODO new method for distributions: logpdf_grad
    push!(stmts, quote
        ($(output_grad_incr, input_grad_incrs...),) = logpdf_grad(
            $dist, $(output_value_ref, input_value_refs...))
        $output_grad_var += $output_grad_incr
    end)
    for (grad_var, incr) in zip(input_grad_vars, input_grad_incrs)
        push!(stmts, quote
            $grad_var += $incr
        end)
    end
    if isa(state.schema, StaticAddressSchema) && addr in leaf_node_keys(state.schema)
        state.values_leaf_nodes[addr] = output_value_ref
        state.gradients_leaf_nodes[addr] = output_grad_var
    end
end

function backprop_trace_process!(state::BBBackpropTraceState, node::AddrGeneratorNode)
    inputs_do_ad = has_argument_grads(node.gen)
    output_do_ad = accepts_output_grad(node.gen)
    input_value_refs = [state.value_refs[in_node] for in_node in node.input_nodes]
    input_grad_vars = [state.grad_vars[in_node] for in_node in node.input_nodes]
    output_value_ref = state.value_refs[node.output]
    output_grad = output_do_ad ? state.grad_vars[node.output] : QuoteNode(nothing)
    addr = node.address
    gen = QuoteNode(node.gen)
    output_grad_incr = gensym("incr")
    input_grad_incrs = [gensym("incr") for _ in node.input_nodes]
    subtrace = Expr(:(.), :trace, QuoteNode(addr))
    has_selection = isa(schema, StaticAddressSchema) && addr in internal_node_keys(schema)
    if has_selection
        selection = Expr(:call, :static_get_internal_node, selection, addr)
        value_trie = gensym("value_trie_$addr")
        gradient_trie = gensym("gradient_trie_$addr")
        state.values_internal_nodes[addr] = value_trie
        state.gradients_internal_nodes[addr] = gradient_trie
    else
        selection = Expr(:call, :EmptyAddressSet)
        value_trie = :_
        gradient_trie = :_
    end
    quote
        (($(input_grad_incrs...),), $value_trie, $gradient_trie) = backprop_trace(
            $gen, $subtrace, $selection, $output_grad)
    end
    for (grad_var, incr, do_ad) in zip(input_grad_vars, input_grad_incrs, inputs_do_ad)
        if do_ad
            push!(stmts, quote
                $grad_var += $incr
            end)
        end
    end
end

function codegen_backprop_trace(gen::Type{T}, trace, selection, retval_grad) where {T <: BasicGenFunction}
    Core.println("generating backprop_trace($gen, selection: $selection...)")
    schema = get_address_schema(constraints)
    ir = get_ir(gen)
    stmts = Expr[]

    # create a gradient variable for each value node, initialize them to zero.
    grad_vars = Dict{ValueNode,Symbol}()
    for (name, node) in ir.value_nodes
        grad_var = gensym("grad_$name")
        grad_vars[node] = grad_var
        if node !== ir.output_node || !ir.output_ad
            # TODO not all of the values that are traced will have a zero() method!
            # should we only compute gradients if they are Float64 or Float32? or subtype of Real?
            # LATER
            push!(stmts, quote
                $grad_var = zero(trace.$(value_field(node)))
            end)
        end
    end

    # initialize the gradient variable for the output node
    if ir.output_node !== nothing && ir.output_ad
        grad_var = grad_vars[ir.output_node]
        push!(stmts, quote
            $grad_var = retval_grad
        end)
    end

    # visit statements in reverse topological order, generating code
    state = BBBackpropTraceState(ir, :trace, stmts, schema, grad_vars)
    for node in reverse(ir.expr_nodes_sorted)
        backprop_trace_process!(state, node)
    end

    # construct values and gradients static choice trie
    values = gensym("values")
    gradients = gensym("gradients")
    (values_leaf_keys, values_leaf_nodes) = collect(zip(state.values_leaf_nodes...))
    (values_internal_keys, values_internal_nodes) = collect(zip(state.values_internal_nodes...))
    (gradients_leaf_keys, gradients_leaf_nodes) = collect(zip(state.gradients_leaf_nodes...))
    (gradients_internal_keys, gradients_internal_nodes) = collect(zip(state.gradients_internal_nodes...))
    push!(stmts, quote
        $values = StaticChoiceTrie(
            NamedTuple{($(values_leaf_keys...),)}(($(values_leaf_nodes...),)),
            NamedTuple{($(values_internal_keys...),)}(($(values_internal_nodes...),)))
        $gradients = StaticChoiceTrie(
            NamedTuple{($(gradients_leaf_keys...),)}(($(gradients_leaf_nodes...),)),
            NamedTuple{($(gradients_internal_keys...),)}(($(gradients_internal_nodes...),)))
    end)

    # gradients with respect to inputs
    input_grads = [has_grad ? grad_var[node] : QuoteNode(nothing) for (node, has_grad) in zip(ir.arg_nodes, ir.args_ad)]

    push!(stmts, quote
        return (($(input_grads...),), $values, $gradients)
    end)
    Expr(:block, stmts...)
end

push!(Gen.generated_functions, quote
@generated function Gen.backprop_trace(gen::Gen.BasicGenFunction, trace, selection, retval_grad)
    schema = get_address_schema(constraints)
    if !(isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema))
        # try to convert it to a static address set
        return quote backprop_trace(gen, trace, StaticAddressSet(selection), retval_grad) end
    end
    Gen.codegen_backprop_trace(gen, trace, selection, retval_grad)
end
end)
