# generating backprop code

# given:
# - a set of selected leaf nodes
# - a set of selection internal nodes 
# - the compute_grad status of the return value

# there are 'source' nodes and 'sink' nodes

# source nodes are the things that we want to differentiate with respect to:
# - any argument node with compute_grad=true
# - any random choice node that is selected
# - (any gen_fn call node that has a nonempty selection and has compute_grad=true) TODO LATER

# sink nodes are things that contribute terms the objective function:
# - the return value node, if it has compute_grad=true
# - the logpdf's of each random choice node
# - (all gen_fn call nodes (but only a subset of inputs may have gradients?) TODO LATER

# we take two passes:

# forward pass:
# - mark the source nodes -- if a random choice node is selected then its *value* is a source
# - take a forward pass through all nodes
#   for each julia node:
#       > if any of its inputs are marked, then mark it
#   for each random choice node:
#       > if any of its inputs are marked, then register it as having LOGPDF on a path from src
#       > (this mark does not propagate)
#   for each gen_fn call node
#       > TODO LATER

# backward pass
# - mark the sink nodes
#       > the return value node is marked
#         if it's a random choice node then the marking refers to the VALUE not the LOGPDF
#       > NOTE XXX: the LOGPDF of ALL random choice nodes are sink nodes - they are implicity marked.
# - take a backward pass through all nodes
#   for each julia node:
#       > if it is marked, then mark all of its parents (marking a random choice means its VALUE)
#   for each random choice node:
#       > regardless of whether the random choice node VALUE is marked, mark
#       all of its parameters (they are on a path to the logpdf, which is a
#       sink)
#       > if the VALUE is marked, this does not change the behavior during the backward pass
#   for each gen_fn call node:

const gradient_prefix = gensym("gradient")
gradient_var(node::RegularNode) = Symbol("$(gradient_prefix)_$(node.name)")

function fwd_pass!(selected_choice_addrs, fwd_marked, fwd_choice_logpdf_marked, node::ArgumentNode)
    if node.compute_grad
        push!(fwd_marked, node)
    end
end

function fwd_pass!(selected_choice_addrs, fwd_marked, fwd_choice_logpdf_marked, node::JuliaNode)
    if any(input_node in fwd_marked for input_node in node.inputs)
        push!(fwd_marked, node)
    end
end

function fwd_pass!(selected_choice_addrs, fwd_marked, fwd_choice_logpdf_marked, node::RandomChoiceNode)
    if node.addr in selected_choice_addrs
        push!(fwd_marked, node) # marking means the VALUE is marked (this propagates)
    end
    if any(input_node in fwd_marked for input_node in node.inputs)
        push!(fwd_choice_logpdf_marked, node)
    end
end

function back_pass!(back_marked, node::ArgumentNode) end

function back_pass!(back_marked, node::JuliaNode)
    if node in back_marked
        for input_node in node.inputs
            push!(back_marked, input_node)
        end
    end
end

function back_pass!(back_marked, node::RandomChoiceNode)
    # the logpdf of every random choice is a SINK
    for input_node in node.inputs
        push!(back_marked, input_node)
    end
    # the value of every random choice is in back_marked, since it affects its logpdf
    push!(back_marked, node) 
end

function fwd_codegen!(stmts, fwd_marked, back_marked, node::ArgumentNode)
    if node in fwd_marked && node in back_marked

        # initialize gradient to zero
        push!(stmts, :($(gradient_var(node)) = zero($(get_value_fieldname(node)))))
    end
end

function fwd_codegen!(stmts, fwd_marked, back_marked, node::JuliaNode)

    # we need the value for initializing gradient to zero (to get the type and
    # e.g. shape), and for reference by other nodes during back_codegen! we
    # could be more selective about which JuliaNodes need to be evalutaed, that
    # is a performance optimization for the future
    args = map((input_node) -> input_node.name, node.inputs)
    push!(stmts, :($(node.name) = $(QuoteNode(node.fn))($(args...))))

    if node in back_marked && any(input_node in fwd_marked for input_node in node.inputs)

        # initialize gradient to zero
        push!(stmts, :($(gradient_var(node)) = zero($(node.name))))
    end
end

function fwd_codegen!(stmts, fwd_marked, back_marked, node::RandomChoiceNode)
    # for reference by other nodes during back_codegen!
    # could performance optimize this away
    push!(stmts, :($(node.name) = $(get_value_fieldname(node))))

    # every random choice is in back_marked, since it affects it logpdf, but
    # also possibly due to other downstream usage of the value
    @assert node in back_marked 

    if node in fwd_marked
        # the only way we are fwd_marked is if this choice was selected

        # initialize gradient with respect to the value of the random choice to zero
        # it will be a runtime error, thrown here, if there is no zero() method
        push!(stmts, :($(gradient_var(node)) = zero($(node.name))))
    end
    
end

function back_codegen!(stmts, ir, fwd_marked, back_marked, node::ArgumentNode)

    # handle case when it is the return node
    if node === ir.return_node && node in fwd_marked
        @assert node in back_marked
        push!(stmts, :($(gradient_var(node)) += retval_grad))
    end
end

function back_codegen!(stmts, ir, fwd_marked, back_marked, node::JuliaNode)
    if node in back_marked && any(input_node in fwd_marked for input_node in node.inputs)

        # compute gradient with respect to parents
        # NOTE: some of the fields in this tuple may be 'nothing'
        input_grads = gensym("input_grads")
        push!(stmts, :($input_grads::Tuple = $(QuoteNode(node.grad_fn))($(gradient_var(node)), $(node.name), $(args...))))

        # increment gradients of input nodes that are in fwd_marked
        for (i, input_node) in enumerate(node.inputs)
            
            # NOTE: it will be a runtime error if we try to add 'nothing'
            # we could require the JuliaNode to statically report which inputs
            # it takes gradients with respect to, and check this at compile
            # time. TODO future work
            if input_node in fwd_marked
                push!(stmts, :($(gradient_var(input_node)) += $input_grads[i]))
            end
        end
    end

    # handle case when it is the return node
    if node === ir.return_node && node in fwd_marked
        @assert node in back_marked
        push!(stmts, :($(gradient_var(node)) += retval_grad))
    end
end

function back_codegen!(stmts, ir, fwd_marked, back_marked, node::RandomChoiceNode)

    # only evaluate the gradient of the logpdf if we need to
    if any(input_node in fwd_marked for input_node in node.inputs) || node in fwd_marked
        logpdf_grad = gensym("logpdf_grad")
        args = map((input_node) -> input_node.name, node.inputs)
        push!(stmts, :($logpdf_grad = logpdf_grad($(node.dist), $(get_value_fieldname(node)), $(args...))))
    end

    # increment gradients of input nodes that are in fwd_marked
    for (i, input_node) in enumerate(node.inputs)
        if input_node in fwd_marked
            @assert input_node in back_marked # this ensured its gradient will have been initialized
            push!(stmts, :($(gradient_var(input_node)) += $logpdf_grad[i+1]))
        end
    end

    # backpropagate to the value (if it was selected)
    if node in fwd_marked
        push!(stmts, :($(gradient_var(node)) += $logpdf_grad[1]))
    end

    # handle case when it is the return node
    if node === ir.return_node && node in fwd_marked
        @assert node in back_marked
        push!(stmts, :($(gradient_var(node)) += retval_grad))
    end
end



function passes()

    # forward marking pass
    # NOTE: if a random choice is in fwd_marked it refers to the VALUE (it is selected)
    fwd_marked = Set{RegularNode}() # these propagate
    fwd_choice_logpdf_marked = Set{RandomChoiceNode}() # means LOGPDF is marked (does not propagate)
    for node in ir.nodes
        fwd_pass!(selected_choice_addrs, fwd_marked, fwd_choice_logpdf_marked, node)
    end

    # backward marking pass
    # NOTE: if a random choice is in back_marked it refers to the VALUE
    back_marked = Set{RegularNode}() # these propagate
    # TODO we should declare compute_return_value_grad as a global property in the
    # IR, not based on the return value node.
    if ir.accepts_return_grad
        push!(back_marked, ir.return_node)
    end
    for node in reverse(ir.nodes)
        back_pass!(back_marked, node)
    end

    # forward code-generation pass (inserts code for JuliaNodes and initializes gradients to zero)
    stmts = Expr[]
    # unpack arguments
    arg_names = Symbol[arg_node.name for arg_node in ir.arg_nodes]
    push!(stmts, :($(Expr(:tuple, arg_names...)) = args))
    for node in ir.nodes
        fwd_codegen!(stmts, fwd_marked, back_marked, node)
    end

    # backward code-generation pass (increment gradients)
    for node in reverse(ir.nodes)
        back_codegen!(stmts, ir, fwd_marked, fwd_choice_logpdf_marked, back_marked, node)
    end
    
    
end



# the nodes in between are:
# - julia nodes that have compute_grad=true/false (if compute_grad=true, a subset of inputs may have gradents)
# - random choices nodes (does compute_grad=true/false matter?)
# - gen_fn call nodes that have compute_grad=true/false (a subset of inputs may have gradients)

# each edge either has a gradient or does not.
# an edge has a gradient if its source node has compute_grad=true and if its
# dest. node has the gradient for that input.



#################################################
# phase 1: do any necessary forward computation #
#################################################

# identify the set of argument nodes for which gradients are needed (based on their annotation)
# identify the set of choice nodes that are selected
# identify whether or not the return value is compute_grad=true/false

# compute the set of julia nodes that need to be evaluated, using a forward/backward pass
# starting from certain marked nodes (arguments with compute_grad=true, and
# selected random choices, and call nodes that have compute_grad=true, and nonempty selection)

# Q: what about nodes that have compute_grad=true and empty selection?
# A: they will propagate the need for gradient in the forward pass, but they aren't sources of it.

# random choices that are not selected do not propagate the need for gradient in the fwd pass

########################################
# phase 2: initialize adjoints to zero #
########################################

# foreach argument node that has compute_grad=true, initialize adjoint of its
# value to zero(current_value)

# for each julia node that has compute_grad=true, initialize adjoint of
# return value to zero(return_value) -- the return value type must have a
# zero() method.

# for each random choice that has compute_value_grad=true, initialize adjoint
# of the return value to zero(return_value) -- the return value type must have
# a zero() method.

# for each generative function call that has compute_value_grad=true (i.e.
# accepts_output_grad), initialize adjoint to zero..


#########################
# phase 3: reverse pass #
#########################

# then, take a backward pass through all the RegularNodes

function codegen_backprop_trace(gen_fn_type::Type{T}, trace_type,
                                selection_type, retval_grad_type) where {T <: StaticIRGenerativeFunctoin}
    schema = get_address_schema(selection_type)
    ir = get_ir(gen_fn_type)

    # phase 1 .. (determine which julia nodes need to be evaluated)
    marked = Set{RegularNode}()
    for node in ir.arg_nodes
        if node.compute_grad
            push!(marked, node)
        end
    end
    if ir.return_node.compute_grad
        push!(marked, ir.return_node)
    end
    #julia_nodes_to_eval = Set{JuliaNode}()
    #call_nodes_to_visit = Set{GenerativeFunctionCallNode}()
    #random_choice_nodes_to_visit = Set{RandomChoiceNode}()
    for node in reverse(ir.nodes)
        # mark a julia node if 
    end
    = get_julia_nodes_to_evaluate(ir, selection_type)

    # phase 2 (initialize gradients to zero)
    stmts = Expr[]
    for node in ir.nodes
        generate_initial_gradient!(stmts, ir, julia_nodes_to_val, node)
    end
    
    # phase 3
    

    # create a gradient variable for each value node, initialize them to zero.
    # also get trace references for each value node
    (value_refs, grad_vars) = initialize_backprop!(ir, stmts)

    # visit statements in reverse topological order
    state = BBBackpropTraceState(gen, :trace, stmts, schema, value_refs, grad_vars)
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





















#############################################
# BBBackpropTraceState (for backprop_trace) #
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


###############################################
# BBBackpropParamsState (for backprop_params) #
###############################################

struct BBBackpropParamsState
    generator_type::Type
    ir::BasicBlockIR
    trace::Symbol
    stmts::Vector{Expr}
    value_refs::Dict{ValueNode,Expr}
    grad_vars::Dict{ValueNode,Symbol}
end

function BBBackpropParamsState(generator_type, trace, stmts, value_refs, grad_vars)
    ir = get_ir(generator_type)
    BBBackpropParamsState(generator_type, ir, trace, stmts, value_refs, grad_vars)
end


#####################################################
# code shared by backprop_trace and backprop_params #
#####################################################

function process!(state::Union{BBBackpropParamsState,BBBackpropTraceState}, node::JuliaNode)
    
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

function process!(state::Union{BBBackpropParamsState,BBBackpropTraceState}, node::ArgsChangeNode)
    # skip
end

function process!(state::Union{BBBackpropParamsState,BBBackpropTraceState}, node::AddrChangeNode)
    # skip
end

function initialize_backprop!(ir::BasicBlockIR, stmts::Vector{Expr})

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

function input_gradients(ir::BasicBlockIR, grad_vars)
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
    if has_output_grad(node.dist)
        if !haskey(grad_vars, node.output)
            error("Distribution $(node.dist) has AD but the return value is not floating point, node: $node")
        end
        output_grad_var = grad_vars[node.output]
        push!(stmts, quote
            $output_grad_var += $increment
        end)
    end
end


###################
# backprop_params #
###################

function process!(state::BBBackpropParamsState, node::AddrDistNode)

    # get gradient of log density with respect to output and inputs
    input_value_refs = [state.value_refs[in_node] for in_node in node.input_nodes]
    output_value_ref = Expr(:(.), :trace, QuoteNode(node.address))# state.value_refs[node.output]
    output_grad_incr = gensym("incr")
    input_grad_incrs = [gensym("incr") for _ in node.input_nodes]
    push!(state.stmts, quote
        ($output_grad_incr, $(input_grad_incrs...),) = Gen.logpdf_grad(
            $(QuoteNode(node.dist)), $output_value_ref, $(input_value_refs...))
    end)

    increment_output_gradient!(state.stmts, node, state.grad_vars, output_grad_incr)
    increment_input_gradients!(state.stmts, node, node.dist, state.grad_vars, input_grad_incrs)
end

function process!(state::BBBackpropParamsState, node::AddrGenerativeFunctionNode)

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

function codegen_backprop_params(gen::Type{T}, trace, retval_grad) where {T <: StaticDSLFunction}
    ir = get_ir(gen)
    stmts = Expr[]

    # create a gradient variable for each value node, initialize them to zero.
    # also get trace references for each value node
    (value_refs, grad_vars) = initialize_backprop!(ir, stmts)

    # visit statements in reverse topological order
    state = BBBackpropParamsState(gen, :trace, stmts, value_refs, grad_vars)
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
@generated function Gen.backprop_params(gen::Gen.StaticDSLFunction, trace, retval_grad)
    Gen.codegen_backprop_params(gen, trace, retval_grad)
end
end)


##################
# backprop_trace #
##################

function process!(state::BBBackpropTraceState, node::AddrDistNode)

    # get gradient of log density with respect to output and inputs
    input_value_refs = [state.value_refs[in_node] for in_node in node.input_nodes]
    output_value_ref = state.value_refs[node.output]
    output_grad_incr = gensym("incr")
    input_grad_incrs = [gensym("incr") for _ in node.input_nodes]
    push!(state.stmts, quote
        ($output_grad_incr, $(input_grad_incrs...),) = Gen.logpdf_grad(
            $(QuoteNode(node.dist)), $output_value_ref, $(input_value_refs...))
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
        push!(state.leaf_nodes, SelectedLeafNode(addr, Expr(:(.), :trace, QuoteNode(addr)), output_grad_var))
    end
end

function process!(state::BBBackpropTraceState, node::AddrGenerativeFunctionNode)

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

function codegen_backprop_trace(gen::Type{T}, trace, selection, retval_grad) where {T <: StaticDSLFunction}
    schema = get_address_schema(selection)
    ir = get_ir(gen)
    stmts = Expr[]

    # create a gradient variable for each value node, initialize them to zero.
    # also get trace references for each value node
    (value_refs, grad_vars) = initialize_backprop!(ir, stmts)

    # visit statements in reverse topological order
    state = BBBackpropTraceState(gen, :trace, stmts, schema, value_refs, grad_vars)
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
@generated function Gen.backprop_trace(gen::Gen.StaticDSLFunction{T,U}, trace::U, selection, retval_grad) where {T,U}
    schema = get_address_schema(selection)
    if !(isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema))
        # try to convert it to a static address set
        return quote backprop_trace(gen, trace, StaticAddressSet(selection), retval_grad) end
    end
    Gen.codegen_backprop_trace(gen, trace, selection, retval_grad)
end
end)
