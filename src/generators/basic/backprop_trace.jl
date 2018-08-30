struct SelectedLeafNode
    addr::Symbol
    value_ref::Expr
    gradient_var::Symbol
end

struct InternalNode
    addr::Symbol
    values_var::Symbol
    gradients_var::Symbol
end

struct BBBackpropTraceState
    generator_type::Type
    ir::BasicBlockIR
    trace::Symbol
    stmts::Vector{Expr}
    schema::Union{StaticAddressSchema,EmptyAddressSchema}
    value_refs::Dict{ValueNode,Expr}
    grad_vars::Dict{ValueNode,Symbol}
    leaf_nodes::Set{SelectedLeafNode}
    internal_nodes::Set{InternalNode}
end

function BBBackpropTraceState(generator_type, trace, stmts, schema, value_refs, grad_vars)
    leaf_nodes = Set{SelectedLeafNode}()
    internal_nodes = Set{InternalNode}()
    ir = get_ir(generator_type)
    BBBackpropTraceState(generator_type, ir, trace, stmts, schema, value_refs, grad_vars,
        leaf_nodes, internal_nodes)
end

function backprop_trace_process!(state::BBBackpropTraceState, node::JuliaNode)
    
    # if the output node is not floating point, don't do anything
    if !haskey(state.grad_vars, node.output)
        return # no statements
    end 

    # call gradient function
    input_value_refs = [state.value_refs[in_node] for in_node in node.input_nodes]
    output_value_ref = state.value_refs[node.output]
    input_grad_increments = [gensym("incr") for _ in node.input_nodes]
    output_grad_var = state.grad_vars[node.output]
    gradient_fn = get_grad_fn(state.generator_type, node)
    push!(state.stmts, quote
        ($(input_grad_increments...),) = $gradient_fn(
            $output_grad_var, $output_value_ref, $(input_value_refs...))
    end)

    # increment input gradients
    inputs_do_ad = map((in_node) -> haskey(state.grad_vars, in_node), node.input_nodes)
    for (in_node, incr, do_ad) in zip(node.input_nodes, input_grad_increments, inputs_do_ad)
        if do_ad
            grad_var = state.grad_vars[in_node]
            push!(state.stmts, quote
                $grad_var += $incr
            end)
        end
    end
end

function backprop_trace_process!(state::BBBackpropTraceState, node::ArgsChangeNode)
    # skip
end

function backprop_trace_process!(state::BBBackpropTraceState, node::AddrChangeNode)
    # skip
end

function backprop_trace_process!(state::BBBackpropTraceState, node::AddrDistNode)

    # NOTE: if the output does AD, then it must be a float...
    # but it may be a float and the grad may not do AD.

    # get gradient of log density with respect to output and inputs
    input_value_refs = [state.value_refs[in_node] for in_node in node.input_nodes]
    output_value_ref = state.value_refs[node.output]
    output_grad_incr = gensym("incr")
    input_grad_incrs = [gensym("incr") for _ in node.input_nodes]
    push!(state.stmts, quote
        ($output_grad_incr, $(input_grad_incrs...),) = Gen.logpdf_grad(
            $(QuoteNode(node.dist)), $output_value_ref, $(input_value_refs...))
    end)

    # increment output gradient
    if has_output_grad(node.dist)
        if !haskey(state.grad_vars, node.output)
            error("Distribution $(node.dist) has AD but the return value is not floating point, node: $node")
        end
        output_grad_var = state.grad_vars[node.output]
        push!(state.stmts, quote
            $output_grad_var += $output_grad_incr
        end)
    end

    # increment input gradients
    inputs_do_ad = has_argument_grads(node.dist)
    for (in_node, do_ad, incr) in zip(node.input_nodes, inputs_do_ad, input_grad_incrs)
        if do_ad
            if !haskey(state.grad_vars, in_node)
                error("Distribution $(node.dist) has AD for an input that is not floating point, node: $node")
            end
            grad_var = state.grad_vars[in_node]
            push!(state.stmts, quote
                $grad_var += $incr
            end)
        end
    end

    # handle selected address
    addr = node.address
    if isa(state.schema, StaticAddressSchema) && addr in leaf_node_keys(state.schema)
        if !haskey(state.grad_vars, node.output)
            error("Selected a random choice that is not floating point for gradient: $node")
        end
        output_grad_var = state.grad_vars[node.output]
        push!(state.leaf_nodes, SelectedLeafNode(addr, Expr(:(.), :trace, QuoteNode(addr)), output_grad_var))
    end
end

function backprop_trace_process!(state::BBBackpropTraceState, node::AddrGeneratorNode)

    # get gradients from generator and handle selection
    output_do_ad = accepts_output_grad(node.gen)
    output_grad = output_do_ad ? state.grad_vars[node.output] : QuoteNode(nothing)
    addr = node.address
    input_grad_incrs = [gensym("incr") for _ in node.input_nodes]
    subtrace = Expr(:(.), :trace, QuoteNode(addr))
    has_selection = isa(state.schema, StaticAddressSchema) && addr in internal_node_keys(state.schema)
    if has_selection
        selection = Expr(:call, :static_get_internal_node, selection, QuoteNode(Val(addr)))
        value_trie = gensym("value_trie_$addr")
        gradient_trie = gensym("gradient_trie_$addr")
        push!(state.internal_nodes, InternalNode(addr, value_trie, gradient_trie))
    else
        selection = quote Gen.EmptyAddressSet() end
        value_trie = :_
        gradient_trie = :_
    end
    push!(state.stmts, quote
        (($(input_grad_incrs...),), $value_trie, $gradient_trie) = backprop_trace(
            $(QuoteNode(node.gen)), $subtrace, $selection, $output_grad)
    end)
    
    # increment input gradients
    inputs_do_ad = has_argument_grads(node.gen)
    for (in_node, do_ad, incr) in zip(node.input_nodes, inputs_do_ad, input_grad_incrs)
        if do_ad
            if !haskey(state.grad_vars, in_node)
                error("Generator $(node.gen) has AD for an input that is not floating point, node: $node")
            end
            grad_var = state.grad_vars[in_node]
            push!(state.stmts, quote
                $grad_var += $incr
            end)
        end
    end
end

function codegen_backprop_trace(gen::Type{T}, trace, selection, retval_grad) where {T <: BasicGenFunction}
    Core.println("generating backprop_trace($gen, selection: $selection...)")
    schema = get_address_schema(selection)
    ir = get_ir(gen)
    stmts = Expr[]

    # create a gradient variable for each value node, initialize them to zero.
    grad_vars = Dict{ValueNode,Symbol}()
    value_refs = Dict{ValueNode,Expr}()
    for (name, node) in ir.value_nodes
        value_refs[node] = value_trace_ref(:trace, node)
        grad_var = gensym("grad_$name")
        if node !== ir.output_node || !ir.output_ad
            if is_differentiable(get_type(node))
                grad_vars[node] = grad_var
                push!(stmts, quote
                    $grad_var = zero($(value_refs[node]))
                end)
            end
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
    state = BBBackpropTraceState(gen, :trace, stmts, schema, value_refs, grad_vars)
    for node in reverse(ir.expr_nodes_sorted)
        backprop_trace_process!(state, node)
    end

    # construct values and gradients static choice tries
    values = gensym("values")
    gradients = gensym("gradients")

    leaf_nodes = collect(state.leaf_nodes)
    quoted_leaf_keys = map((node) -> QuoteNode(node.addr), leaf_nodes)
    leaf_values = map((node) -> node.value_ref, leaf_nodes)
    leaf_gradients = map((node) -> node.gradient_var, leaf_nodes)

    internal_nodes = collect(state.internal_nodes)
    quoted_internal_keys = map((node) -> QuoteNode(node.addr), internal_nodes)
    internal_values = map((node) -> node.values_var, internal_nodes)
    internal_gradients = map((node) -> node.gradients_var, internal_nodes)

    push!(stmts, quote
        $values = StaticChoiceTrie(
            NamedTuple{($(quoted_leaf_keys...),)}(($(leaf_values...),)),
            NamedTuple{($(quoted_internal_keys...),)}(($(internal_values...),)))
        $gradients = StaticChoiceTrie(
            NamedTuple{($(quoted_leaf_keys...),)}(($(leaf_gradients...),)),
            NamedTuple{($(quoted_internal_keys...),)}(($(internal_gradients...),)))
    end)

    # gradients with respect to inputs
    input_grads_var = gensym("input_grads")
    input_grads = []
    for (node, has_grad) in zip(ir.arg_nodes, ir.args_ad)
        if has_grad
            push!(input_grads, grad_vars[node])
        else
            push!(input_grads, QuoteNode(nothing))
        end
    end
    push!(stmts, quote
        $input_grads_var = ($(input_grads...),)
    end)
    push!(stmts, quote
        return ($input_grads_var, $values, $gradients)
    end)
    Expr(:block, stmts...)
end

push!(Gen.generated_functions, quote
@generated function Gen.backprop_trace(gen::Gen.BasicGenFunction, trace, selection, retval_grad)
    schema = get_address_schema(selection)
    if !(isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema))
        # try to convert it to a static address set
        return quote backprop_trace(gen, trace, StaticAddressSet(selection), retval_grad) end
    end
    Gen.codegen_backprop_trace(gen, trace, selection, retval_grad)
end
end)
