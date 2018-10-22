#############################################
# StaticDataFlowBackpropTraceState (for backprop_trace) #
#############################################

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

struct StaticDataFlowBackpropTraceState
    generator_type::Type
    ir::DataFlowIR
    trace::Symbol
    stmts::Vector{Expr}
    schema::Union{StaticAddressSchema,EmptyAddressSchema}
    value_refs::Dict{ValueNode,Expr}
    grad_vars::Dict{ValueNode,Symbol}
    leaf_nodes::Set{SelectedLeafNode}
    internal_nodes::Set{InternalNode}
end

function StaticDataFlowBackpropTraceState(generator_type, trace, stmts, schema, value_refs, grad_vars)
    leaf_nodes = Set{SelectedLeafNode}()
    internal_nodes = Set{InternalNode}()
    ir = get_ir(generator_type)
    StaticDataFlowBackpropTraceState(generator_type, ir, trace, stmts, schema, value_refs, grad_vars,
        leaf_nodes, internal_nodes)
end


###########################################################
# StaticDataFlowBackpropParamsState (for backprop_params) #
###########################################################

struct StaticDataFlowBackpropParamsState
    generator_type::Type
    ir::DataFlowIR
    trace::Symbol
    stmts::Vector{Expr}
    value_refs::Dict{ValueNode,Expr}
    grad_vars::Dict{ValueNode,Symbol}
end

function StaticDataFlowBackpropParamsState(generator_type, trace, stmts, value_refs, grad_vars)
    ir = get_ir(generator_type)
    StaticDataFlowBackpropParamsState(generator_type, ir, trace, stmts, value_refs, grad_vars)
end


#####################################################
# code shared by backprop_trace and backprop_params #
#####################################################

function process!(state::Union{StaticDataFlowBackpropParamsState,StaticDataFlowBackpropTraceState}, node::JuliaNode)
    
    # if the output node is not floating point, don't do anything
    if !haskey(state.grad_vars, node.output)
        return # no statements
    end 

    # call gradient function
    inputs = [state.value_refs[in_node] for in_node in node.input_nodes]
    output = state.value_refs[node.output]
    input_grad_increments = [gensym("incr") for _ in node.input_nodes]
    output_grad_var = state.grad_vars[node.output]
    gradient_fn = get_grad_fn(state.generator_type, node)
    push!(state.stmts, quote
        ($(input_grad_increments...),) = $gradient_fn(
            $output_grad_var, $output, $(inputs...))
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

function process!(state::Union{StaticDataFlowBackpropParamsState,StaticDataFlowBackpropTraceState}, node::ArgsChangeNode)
    # skip
end

function process!(state::Union{StaticDataFlowBackpropParamsState,StaticDataFlowBackpropTraceState}, node::AddrChangeNode)
    # skip
end

function initialize_backprop!(ir::DataFlowIR, stmts::Vector{Expr})

    # create a gradient variable for each value node, initialize them to zero.
    value_refs = Dict{ValueNode,Expr}()
    grad_vars = Dict{ValueNode,Symbol}()
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
        grad_var = grad_vars[something(ir.output_node)]
        push!(stmts, quote
            $grad_var = retval_grad
        end)
    end

    (value_refs, grad_vars)
end

function input_gradients(ir::DataFlowIR, grad_vars)
    input_grads_var = gensym("input_grads")
    input_grads = []
    for (node, has_grad) in zip(ir.arg_nodes, ir.args_ad)
        if has_grad
            push!(input_grads, grad_vars[node])
        else
            push!(input_grads, QuoteNode(nothing))
        end
    end
    Expr(:tuple, input_grads...)
end

function increment_input_gradients!(stmts, node, dist_or_gen, grad_vars, increments)
    inputs_do_ad = has_argument_grads(dist_or_gen)
    for (in_node, do_ad, incr) in zip(node.input_nodes, inputs_do_ad, increments)
        if do_ad
            if !haskey(grad_vars, in_node)
                error("$(dist_or_gen) has AD for an input that is not floating point, node: $node")
            end
            grad_var = grad_vars[in_node]
            push!(stmts, quote
                $grad_var += $incr
            end)
        end
    end
end

    
function increment_output_gradient!(stmts, node::AddrDistNode, grad_vars, increment)
    # NOTE: if the output has a gradient, then it must be a float...
    # but it may be a float and the may not hae a gradient, currently this is
    # silent; could warn?
    if !haskey(grad_vars, node.output)
        error("Distribution $(node.dist) has AD but the return value is not floating point, node: $node")
    end
    output_grad_var = grad_vars[node.output]
    push!(stmts, quote
        $output_grad_var += $increment
    end)
end


###################
# backprop_params #
###################

function process!(state::StaticDataFlowBackpropParamsState, node::AddrDistNode)

    # get gradient of log density with respect to output and inputs
    inputs = [state.value_refs[in_node] for in_node in node.input_nodes]
    output = state.value_refs[node.output]
    output_grad_incr = gensym("incr")
    input_grad_incrs = [gensym("incr") for _ in node.input_nodes]
    push!(state.stmts, quote
        ($output_grad_incr, $(input_grad_incrs...),) = Gen.logpdf_grad(
            $(QuoteNode(node.dist)), $output, $(inputs...))
    end)
    increment_output_gradient!(state.stmts, node, state.grad_vars, output_grad_incr)
    increment_input_gradients!(state.stmts, node, node.dist, state.grad_vars, input_grad_incrs)
end

function process!(state::StaticDataFlowBackpropParamsState, node::AddrGeneratorNode)

    # get gradients from generator 
    output_do_ad = accepts_output_grad(node.gen)
    output_grad = output_do_ad ? state.grad_vars[node.output] : QuoteNode(nothing)
    addr = node.address
    input_grad_incrs = [gensym("incr") for _ in node.input_nodes]
    subtrace = Expr(:(.), :trace, QuoteNode(addr))
    push!(state.stmts, quote
        ($(input_grad_incrs...),) = backprop_params($(QuoteNode(node.gen)), $subtrace, $output_grad)
    end)
    
    increment_input_gradients!(state.stmts, node, node.gen, state.grad_vars, input_grad_incrs)
end

function codegen_backprop_params(gen::Type{T}, trace, retval_grad) where {T <: StaticDataFlowGenerator}
    ir = get_ir(gen)
    stmts = Expr[]

    # create a gradient variable for each value node, initialize them to zero.
    # also get trace references for each value node
    (value_refs, grad_vars) = initialize_backprop!(ir, stmts)

    # visit statements in reverse topological order
    state = StaticDataFlowBackpropParamsState(gen, :trace, stmts, value_refs, grad_vars)
    for node in reverse(ir.expr_nodes_sorted)
        process!(state, node)
    end

    # increment gradient accumulators for parameters
    for param in ir.params
        value_node = ir.value_nodes[param.name]
        grad_var = grad_vars[value_node]
        push!(stmts, quote
            gen.params_grad[$(QuoteNode(param.name))] += $grad_var
        end)
    end

    # return statement
    push!(stmts, quote
        return $(input_gradients(ir, grad_vars))
    end)
    Expr(:block, stmts...)
end

push!(Gen.generated_functions, quote
@generated function Gen.backprop_params(gen::Gen.StaticDataFlowGenerator, trace, retval_grad)
    Gen.codegen_backprop_params(gen, trace, retval_grad)
end
end)


##################
# backprop_trace #
##################

function process!(state::StaticDataFlowBackpropTraceState, node::AddrDistNode)

    # get gradient of log density with respect to output and inputs
    inputs = [state.value_refs[in_node] for in_node in node.input_nodes]
    output = state.value_refs[node.output]
    output_grad_incr = gensym("incr")
    input_grad_incrs = [gensym("incr") for _ in node.input_nodes]
    push!(state.stmts, quote
        ($output_grad_incr, $(input_grad_incrs...),) = Gen.logpdf_grad(
            $(QuoteNode(node.dist)), $output, $(inputs...))
    end)

    increment_output_gradient!(state.stmts, node, state.grad_vars, output_grad_incr)
    increment_input_gradients!(state.stmts, node, node.dist, state.grad_vars, input_grad_incrs)

    # handle selected address
    addr = node.address
    if isa(state.schema, StaticAddressSchema) && addr in leaf_node_keys(state.schema)
        if !haskey(state.grad_vars, node.output)
            error("Selected a random choice that is not floating point for gradient: $node")
        end
        output_grad_var = state.grad_vars[node.output]
        push!(state.leaf_nodes, SelectedLeafNode(addr, value, output_grad_var))
    end
end

function process!(state::StaticDataFlowBackpropTraceState, node::AddrGeneratorNode)

    # get gradients from generator and handle selection
    output_do_ad = accepts_output_grad(node.gen)
    output_grad = output_do_ad ? state.grad_vars[node.output] : QuoteNode(nothing)
    addr = node.address
    input_grad_incrs = [gensym("incr") for _ in node.input_nodes]
    subtrace = Expr(:(.), :trace, QuoteNode(addr))
    has_selection = isa(state.schema, StaticAddressSchema) && addr in internal_node_keys(state.schema)
    if has_selection
        selection = Expr(:call, :static_get_internal_node, :selection, QuoteNode(Val(addr)))
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
    
    increment_input_gradients!(state.stmts, node, node.gen, state.grad_vars, input_grad_incrs)
end

const backprop_values_trie = gensym("values")
const backprop_gradients_trie = gensym("gradients")

function choice_trie_construction(leaf_nodes_set, internal_nodes_set)
    leaf_nodes = collect(leaf_nodes_set)
    quoted_leaf_keys = map((node) -> QuoteNode(node.addr), leaf_nodes)
    leaf_values = map((node) -> node.value_ref, leaf_nodes)
    leaf_gradients = map((node) -> node.gradient_var, leaf_nodes)
    internal_nodes = collect(internal_nodes_set)
    quoted_internal_keys = map((node) -> QuoteNode(node.addr), internal_nodes)
    internal_values = map((node) -> node.values_var, internal_nodes)
    internal_gradients = map((node) -> node.gradients_var, internal_nodes)
    quote
        $backprop_values_trie = StaticAssignment(
            NamedTuple{($(quoted_leaf_keys...),)}(($(leaf_values...),)),
            NamedTuple{($(quoted_internal_keys...),)}(($(internal_values...),)))
        $backprop_gradients_trie = StaticAssignment(
            NamedTuple{($(quoted_leaf_keys...),)}(($(leaf_gradients...),)),
            NamedTuple{($(quoted_internal_keys...),)}(($(internal_gradients...),)))
    end
end

function codegen_backprop_trace(gen::Type{T}, trace, selection, retval_grad) where {T <: StaticDataFlowGenerator}
    schema = get_address_schema(selection)
    ir = get_ir(gen)
    stmts = Expr[]

    # create a gradient variable for each value node, initialize them to zero.
    # also get trace references for each value node
    (value_refs, grad_vars) = initialize_backprop!(ir, stmts)

    # visit statements in reverse topological order
    state = StaticDataFlowBackpropTraceState(gen, :trace, stmts, schema, value_refs, grad_vars)
    for node in reverse(ir.expr_nodes_sorted)
        process!(state, node)
    end

    # construct values and gradients static choice tries
    push!(stmts, choice_trie_construction(state.leaf_nodes, state.internal_nodes))

    # return statement
    push!(stmts, quote
        return ($(input_gradients(ir, grad_vars)), $backprop_values_trie, $backprop_gradients_trie)
    end)
    Expr(:block, stmts...)
end

push!(Gen.generated_functions, quote
@generated function Gen.backprop_trace(gen::Gen.StaticDataFlowGenerator{T,U}, trace::U, selection, retval_grad) where {T,U}
    schema = get_address_schema(selection)
    if !(isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema))
        # try to convert it to a static address set
        return quote backprop_trace(gen, trace, StaticAddressSet(selection), retval_grad) end
    end
    Gen.codegen_backprop_trace(gen, trace, selection, retval_grad)
end
end)
