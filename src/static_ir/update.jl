abstract type UpdateMode end
struct ForceUpdateMode <: UpdateMode end
struct ExtendMode <: UpdateMode end


"""
Example: MaskedArgDiff{Tuple{true, false, true}, Int}(5)
"""
struct MaskedArgDiff{T<:Tuple,U}
    argdiff::U
end

export MaskedArgDiff

const choicediff_prefix = gensym("choicediff")
choicediff_var(node::RandomChoiceNode) = Symbol("$(choicediff_prefix)_$(node.addr)")

const calldiff_prefix = gensym("calldiff")
calldiff_var(node::GenerativeFunctionCallNode) = Symbol("$(calldiff_prefix)_$(node.addr)")

const choice_discard_prefix = gensym("choice_discard")
choice_discard_var(node::RandomChoiceNode) = Symbol("$(choice_discard_prefix)_$(node.addr)")

const call_discard_prefix = gensym("call_discard")
call_discard_var(node::GenerativeFunctionCallNode) = Symbol("$(call_discard_prefix)_$(node.addr)")

struct ForwardPassState
    input_changed::Set{Union{RandomChoiceNode,GenerativeFunctionCallNode}}
    value_changed::Set{RegularNode}
    constrained_choices::Set{RandomChoiceNode} # the same as discarded choices
    constrained_calls::Set{GenerativeFunctionCallNode}
    discard_calls::Set{GenerativeFunctionCallNode}
end

function ForwardPassState()
    input_changed = Set{Union{RandomChoiceNode,GenerativeFunctionCallNode}}()
    value_changed = Set{RegularNode}()
    constrained_choices = Set{RandomChoiceNode}()
    constrained_calls = Set{GenerativeFunctionCallNode}()
    discard_calls = Set{GenerativeFunctionCallNode}()
    ForwardPassState(input_changed, value_changed, constrained_choices,
        constrained_calls, discard_calls)
end

struct BackwardPassState 
    marked::Set{StaticIRNode}
end

function BackwardPassState()
    BackwardPassState(Set{StaticIRNode}())
end

function forward_pass_argdiff!(state::ForwardPassState,
                               arg_nodes::Vector{ArgumentNode},
                               ::Type{UnknownArgDiff})
    for node in arg_nodes
        push!(state.value_changed, node)
    end
end

function forward_pass_argdiff!(state::ForwardPassState,
                               arg_nodes::Vector{ArgumentNode},
                               ::Type{NoArgDiff})
    for node in arg_nodes
        push!(state.value_changed, node)
    end
end

function forward_pass_argdiff!(state::ForwardPassState,
                               arg_nodes::Vector{ArgumentNode},
                               ::Type{MaskedArgDiff{T,U}}) where {T<:Tuple,U}
    for (node, marked::Bool) in zip(arg_nodes, T.parameters)
        push!(state.value_changed, node)
    end
end

function process_forward!(::AddressSchema, ::ForwardPassState, ::ArgumentNode) end
function process_forward!(::AddressSchema, ::ForwardPassState, ::DiffJuliaNode) end
function process_forward!(::AddressSchema, ::ForwardPassState, ::ReceivedArgDiffNode) end
function process_forward!(::AddressSchema, ::ForwardPassState, ::ChoiceDiffNode) end
function process_forward!(::AddressSchema, ::ForwardPassState, ::CallDiffNode) end

function process_forward!(::AddressSchema, state::ForwardPassState, node::JuliaNode)
    if any(input_node in state.value_changed for input_node in node.inputs)
        push!(state.value_changed, node)
    end
end

function process_forward!(schema::AddressSchema, state::ForwardPassState,
                          node::RandomChoiceNode)
    @assert isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema)
    if isa(schema, StaticAddressSchema) && (node.addr in leaf_node_keys(schema))
        push!(state.constrained_choices, node)
        push!(state.value_changed, node)
    end
    if any(input_node in state.value_changed for input_node in node.inputs)
        push!(state.input_changed, node)
    end
end

function process_forward!(schema::AddressSchema, state::ForwardPassState,
                          node::GenerativeFunctionCallNode)
    @assert isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema)
    if isa(schema, StaticAddressSchema) && (node.addr in internal_node_keys(schema))
        push!(state.constrained_calls, node)
        push!(state.value_changed, node)
        push!(state.discard_calls, node)
    end
    if any(input_node in state.value_changed for input_node in node.inputs)
        push!(state.input_changed, node)
        push!(state.value_changed, node)
        push!(state.discard_calls, node)
    end
end

function process_backward!(::ForwardPassState, ::BackwardPassState, ::ArgumentNode) end

function process_backward!(::ForwardPassState, back::BackwardPassState, node::DiffJuliaNode)
    if node in back.marked
        for input_node in node.inputs
            push!(back.marked, input_node)
        end
    end
end

function process_backward!(::ForwardPassState, ::BackwardPassState, ::ReceivedArgDiffNode) end
function process_backward!(::ForwardPassState, ::BackwardPassState, ::ChoiceDiffNode) end
function process_backward!(::ForwardPassState, ::BackwardPassState, ::CallDiffNode) end

function process_backward!(::ForwardPassState, back::BackwardPassState,
                           node::JuliaNode)
    if node in back.marked
        for input_node in node.inputs
            push!(back.marked, input_node)
        end
    end
end

function process_backward!(fwd::ForwardPassState, back::BackwardPassState,
                           node::RandomChoiceNode)
    if node in fwd.input_changed || node in fwd.constrained_choices
        for input_node in node.inputs
            push!(back.marked, input_node)
        end
    end
end

function process_backward!(fwd::ForwardPassState, back::BackwardPassState,
                           node::GenerativeFunctionCallNode)
    if node in fwd.input_changed || node in fwd.constrained_calls
        for input_node in node.inputs
            push!(back.marked, input_node)
        end
        push!(back.marked, node.argdiff)
    end
end

function process_codegen!(stmts, ::ForwardPassState, ::BackwardPassState,
                          node::ArgumentNode, ::UpdateMode)
    push!(stmts, :($(get_value_fieldname(node)) = $(node.name)))
end

function process_codegen!(stmts, ::ForwardPassState, back::BackwardPassState,
                          node::DiffJuliaNode, ::UpdateMode)
    if node in back.marked
        args = map((input_node) -> input_node.name, node.inputs)
        push!(stmts, :($(node.name) = $(QuoteNode(node.fn))($(args...))))
    end
end

function process_codegen!(stmts, ::ForwardPassState, back::BackwardPassState,
                          node::ReceivedArgDiffNode, ::UpdateMode)
    if node in back.marked
        push!(stmts, :($(node.name) = argdiff))
    end
end

function process_codegen!(stmts, fwd::ForwardPassState, back::BackwardPassState,
                          node::ChoiceDiffNode, ::UpdateMode)
    if node in back.marked
        if node.choice_node in fwd.constrained_choices || node.choice_node in fwd.input_changed
            push!(stmts, :($(node.name) = $(choicediff_var(node.choice_node))))
        else
            push!(stmts, :($(node.name) = NoChoiceDiff()))
        end
    end
end

function process_codegen!(stmts, fwd::ForwardPassState, back::BackwardPassState,
                          node::CallDiffNode, ::UpdateMode)
    if node in back.marked
        if node.call_node in fwd.constrained_calls || node.call_node in fwd.input_changed
            push!(stmts, :($(node.name) = $(calldiff_var(node.call_node))))
        else
            push!(stmts, :($(node.name) = NoCallDiff()))
        end
    end
end

function process_codegen!(stmts, fwd::ForwardPassState, back::BackwardPassState,
                         node::JuliaNode, ::UpdateMode)
    if node in back.marked
        args = map((input_node) -> input_node.name, node.inputs)
        push!(stmts, :($(node.name) = $(QuoteNode(node.fn))($(args...))))
    end
end

function process_codegen!(stmts, fwd::ForwardPassState, back::BackwardPassState,
                          node::RandomChoiceNode, ::UpdateMode)
    args = map((input_node) -> input_node.name, node.inputs)
    new_logpdf = gensym("new_logpdf")
    addr = QuoteNode(node.addr)
    dist = QuoteNode(node.dist)
    if node in fwd.constrained_choices || node in fwd.input_changed
        if node in fwd.constrained_choices
            push!(stmts, :($(node.name) = static_get_value(constraints, Val($addr))))
            push!(stmts, :($(choice_discard_var(node)) = trace.$(get_value_fieldname(node))))
        else
            push!(stmts, :($(node.name) = trace.$(get_value_fieldname(node))))
        end
        push!(stmts, :($new_logpdf = logpdf($dist, $(node.name), $(args...))))
        push!(stmts, :($weight += $new_logpdf - trace.$(get_score_fieldname(node))))
        #push!(stmts, :($total_score_fieldname += $new_logpdf - trace.$(get_score_fieldname(node))))
        push!(stmts, :($(get_score_fieldname(node)) = $new_logpdf))
        push!(stmts, :($(choicediff_var(node)) = PrevChoiceDiff(trace.$(get_value_fieldname(node)))))
    else
        push!(stmts, :($(node.name) = trace.$(get_value_fieldname(node))))
        push!(stmts, :($(get_score_fieldname(node)) = trace.$(get_score_fieldname(node))))
    end
    push!(stmts, :($(get_value_fieldname(node)) = $(node.name)))
end

function process_codegen!(stmts, fwd::ForwardPassState, back::BackwardPassState,
                          node::GenerativeFunctionCallNode, ::ForceUpdateMode)
    args = map((input_node) -> input_node.name, node.inputs)
    args_tuple = Expr(:tuple, args...)
    addr = QuoteNode(node.addr)
    gen_fn = QuoteNode(node.generative_function)
    subtrace = get_subtrace_fieldname(node)
    prev_subtrace = :(trace.$subtrace)
    call_weight = gensym("call_weight")
    call_constraints = gensym("call_constraints")
    if node in fwd.constrained_calls || node in fwd.input_changed
        if node in fwd.constrained_calls
            push!(stmts, :($call_constraints = static_get_subassmt(constraints, Val($addr))))
        else
            push!(stmts, :($call_constraints = EmptyAssignment()))
        end
        push!(stmts, :(($subtrace, $call_weight, $(call_discard_var(node)), $(calldiff_var(node))) = 
            force_update($gen_fn, $args_tuple, $(node.argdiff.name), $(prev_subtrace), $call_constraints)
        ))
        push!(stmts, :($weight += $call_weight))
        #push!(stmts, :($total_score_fieldname += $call_weight))
        push!(stmts, :(if has_choices($subtrace) && !has_choices($prev_subtrace)
                            $num_has_choices_fieldname += 1 end))
        push!(stmts, :(if !has_choices($subtrace) && has_choices($prev_subtrace)
                            $num_has_choices_fieldname -= 1 end))
    else
        push!(stmts, :($subtrace = $prev_subtrace))
    end
    push!(stmts, :($(node.name) = get_retval($subtrace)))
end

function process_codegen!(stmts, fwd::ForwardPassState, back::BackwardPassState,
                          node::GenerativeFunctionCallNode, ::ExtendMode)
    args = map((input_node) -> input_node.name, node.inputs)
    args_tuple = Expr(:tuple, args...)
    addr = QuoteNode(node.addr)
    gen_fn = QuoteNode(node.generative_function)
    subtrace = get_subtrace_fieldname(node)
    prev_subtrace = :(trace.$subtrace)
    call_weight = gensym("call_weight")
    call_constraints = gensym("call_constraints")
    if node in fwd.constrained_calls || node in fwd.input_changed
        if node in fwd.constrained_calls
            push!(stmts, :($call_constraints = static_get_subassmt(constraints, Val($addr))))
        else
            push!(stmts, :($call_constraints = EmptyAssignment()))
        end
        push!(stmts, :(($subtrace, $call_weight, $(calldiff_var(node))) = 
            extend($gen_fn, $args_tuple, $(node.argdiff.name), $(prev_subtrace), $call_constraints)
        ))
        push!(stmts, :($weight += $call_weight))
        push!(stmts, :($total_score_fieldname += $call_weight))
        push!(stmts, :(if has_choices($subtrace) && !has_choices($prev_subtrace)
                            $num_has_choices_fieldname += 1 end))
        push!(stmts, :(if !has_choices($subtrace) && has_choices($prev_subtrace)
                            $num_has_choices_fieldname -= 1 end))
    else
        push!(stmts, :($subtrace = $prev_subtrace))
    end
    push!(stmts, :($(node.name) = get_retval($subtrace)))
end

function initialize_score_weight_num_has_choices!(stmts::Vector{Expr})
    #push!(stmts, :($total_score_fieldname = trace.$total_score_fieldname))
    push!(stmts, :($weight = 0.))
    push!(stmts, :($num_has_choices_fieldname = trace.$num_has_choices_fieldname))
end

function unpack_arguments!(stmts::Vector{Expr}, arg_nodes::Vector{ArgumentNode})
    arg_names = Symbol[arg_node.name for arg_node in arg_nodes]
    push!(stmts, :($(Expr(:tuple, arg_names...)) = args))
end

function generate_return_value!(stmts::Vector{Expr}, return_node::RegularNode)
    push!(stmts, :($return_value_fieldname = $(return_node.name)))
end

function generate_new_trace!(stmts::Vector{Expr}, trace_type::Type)
    push!(stmts, :($trace = $(QuoteNode(trace_type))($(fieldnames(trace_type)...))))
end

function generate_retdiff!(stmts::Vector{Expr}, retdiff_node::StaticIRNode)
    push!(stmts, :($retdiff = $(retdiff_node.name)))
end

function generate_discard!(stmts::Vector{Expr},
                           constrained_choices::Set{RandomChoiceNode},
                           discard_calls::Set{GenerativeFunctionCallNode})
    discard_leaf_nodes = Dict{Symbol,Symbol}()
    for node in constrained_choices
        discard_leaf_nodes[node.addr] = choice_discard_var(node)
    end
    discard_internal_nodes = Dict{Symbol,Symbol}()
    for node in discard_calls
        discard_internal_nodes[node.addr] = call_discard_var(node)
    end
    if length(discard_leaf_nodes) > 0
        (leaf_keys, leaf_nodes) = collect(zip(discard_leaf_nodes...))
    else
        (leaf_keys, leaf_nodes) = ((), ())
    end
    if length(discard_internal_nodes) > 0
        (internal_keys, internal_nodes) = collect(zip(discard_internal_nodes...))
    else
        (internal_keys, internal_nodes) = ((), ())
    end
    leaf_keys = map((key::Symbol) -> QuoteNode(key), leaf_keys)
    internal_keys = map((key::Symbol) -> QuoteNode(key), internal_keys)
    expr = :(StaticAssignment(
            NamedTuple{($(leaf_keys...),)}(($(leaf_nodes...),)),
            NamedTuple{($(internal_keys...),)}(($(internal_nodes...),))))
    push!(stmts, :($discard = $expr))
end

function codegen_force_update(gen_fn_type::Type{G}, args_type, argdiff_type,
                              trace_type::Type{U}, constraints_type) where {T,U,G<:StaticIRGenerativeFunction{T,U}}
    schema = get_address_schema(constraints_type)

    # convert the constraints to a static assignment if it is not already one
    if !(isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema))
        return quote force_update(gen_fn, args, argdiff, trace, StaticAssignment(constraints)) end
    end

    ir = get_ir(gen_fn_type)

    # forward marking pass
    forward_state = ForwardPassState()
    forward_pass_argdiff!(forward_state, ir.arg_nodes, argdiff_type)
    for node in ir.nodes
        process_forward!(schema, forward_state, node)
    end

    # backward marking pass
    backward_state = BackwardPassState()
    push!(backward_state.marked, ir.return_node)
    push!(backward_state.marked, ir.retdiff_node)
    for node in reverse(ir.nodes)
        process_backward!(forward_state, backward_state, node)
    end

    # forward code generation pass
    stmts = Expr[]
    initialize_score_weight_num_has_choices!(stmts)
    unpack_arguments!(stmts, ir.arg_nodes)
    for node in ir.nodes
        process_codegen!(stmts, forward_state, backward_state, node, ForceUpdateMode())
    end
    generate_return_value!(stmts, ir.return_node)
    generate_new_trace!(stmts, trace_type)
    generate_discard!(stmts, forward_state.constrained_choices, forward_state.discard_calls)
    generate_retdiff!(stmts, ir.retdiff_node)

    # return trace and weight and discard
    push!(stmts, :(return ($trace, $weight, $discard, $retdiff)))

    Expr(:block, stmts...)
end

function codegen_extend(gen_fn_type::Type{G}, args_type, argdiff_type,
                        trace_type::Type{U}, constraints_type) where {T,U,G<:StaticIRGenerativeFunction{T,U}}
    schema = get_address_schema(constraints_type)

    # convert the constraints to a static assignment if it is not already one
    if !(isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema))
        return quote extend(gen_fn, args, argdiff, trace, StaticAssignment(constraints)) end
    end

    ir = get_ir(gen_fn_type)

    # forward marking pass
    forward_state = ForwardPassState()
    forward_pass_argdiff!(forward_state, ir.arg_nodes, argdiff_type)
    for node in ir.nodes
        process_forward!(schema, forward_state, node)
    end

    # backward marking pass
    backward_state = BackwardPassState()
    push!(backward_state.marked, ir.return_node)
    push!(backward_state.marked, ir.retdiff_node)
    for node in reverse(ir.nodes)
        process_backward!(forward_state, backward_state, node)
    end

    # forward code generation pass
    stmts = Expr[]
    initialize_score_weight_num_has_choices!(stmts)
    unpack_arguments!(stmts, ir.arg_nodes)
    for node in ir.nodes
        process_codegen!(stmts, forward_state, backward_state, node, ExtendMode())
    end
    generate_return_value!(stmts, ir.return_node)
    generate_new_trace!(stmts, trace_type)
    generate_retdiff!(stmts, ir.retdiff_node)

    # return trace and weight and discard
    push!(stmts, :(return ($trace, $weight, $retdiff)))

    Expr(:block, stmts...)
end

push!(Gen.generated_functions, quote
@generated function Gen.force_update(gen_fn::Gen.StaticIRGenerativeFunction{T,U}, args::Tuple,
                                     argdiff::Union{NoArgDiff,UnknownArgDiff,MaskedArgDiff},
                                     trace::U, constraints::Assignment) where {T,U}
    Gen.codegen_force_update(gen_fn, args, argdiff, trace, constraints)
end
end)

push!(Gen.generated_functions, quote
@generated function Gen.extend(gen_fn::Gen.StaticIRGenerativeFunction{T,U}, args::Tuple,
                               argdiff::Union{NoArgDiff,UnknownArgDiff,MaskedArgDiff},
                               trace::U, constraints::Assignment) where {T,U}
    Gen.codegen_extend(gen_fn, args, argdiff, trace, constraints)
end
end)
