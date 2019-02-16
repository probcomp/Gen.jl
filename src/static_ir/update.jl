abstract type AbstractUpdateMode end
struct UpdateMode <: AbstractUpdateMode end
struct ExtendMode <: AbstractUpdateMode end
struct RegenerateMode <: AbstractUpdateMode end

const retdiff = gensym("retdiff")
const discard = gensym("discard")

const calldiff_prefix = gensym("calldiff")
calldiff_var(node::GenerativeFunctionCallNode) = Symbol("$(calldiff_prefix)_$(node.addr)")

const choice_discard_prefix = gensym("choice_discard")
choice_discard_var(node::RandomChoiceNode) = Symbol("$(choice_discard_prefix)_$(node.addr)")

const call_discard_prefix = gensym("call_discard")
call_discard_var(node::GenerativeFunctionCallNode) = Symbol("$(call_discard_prefix)_$(node.addr)")

########################
# forward marking pass #
########################

struct ForwardPassState
    input_changed::Set{Union{RandomChoiceNode,GenerativeFunctionCallNode}}
    value_changed::Set{StaticIRNode}
    constrained_or_selected_choices::Set{RandomChoiceNode}
    constrained_or_selected_calls::Set{GenerativeFunctionCallNode}
    discard_calls::Set{GenerativeFunctionCallNode}
end

function ForwardPassState()
    input_changed = Set{Union{RandomChoiceNode,GenerativeFunctionCallNode}}()
    value_changed = Set{StaticIRNode}()
    constrained_or_selected_choices = Set{RandomChoiceNode}()
    constrained_or_selected_calls = Set{GenerativeFunctionCallNode}()
    discard_calls = Set{GenerativeFunctionCallNode}()
    ForwardPassState(input_changed, value_changed, constrained_or_selected_choices,
        constrained_or_selected_calls, discard_calls)
end

function forward_pass_argdiff!(state::ForwardPassState,
                               arg_nodes::Vector{ArgumentNode},
                               argdiffs_type::Type)
    for (node, diff_typ) in zip(arg_nodes, argdiffs_type.parameters)
        push!(state.valued_changed, node)
    end
end

function process_forward!(::AddressSchema, ::ForwardPassState, node) end

function process_forward!(::AddressSchema, state::ForwardPassState, node::JuliaNode)
    if any(input_node in state.value_changed for input_node in node.inputs)
        push!(state.value_changed, node)
    end
end

function process_forward!(schema::AddressSchema, state::ForwardPassState,
                          node::RandomChoiceNode)
    @assert isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema)
    if isa(schema, StaticAddressSchema) && (node.addr in leaf_node_keys(schema))
        push!(state.constrained_or_selected_choices, node)
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
        push!(state.constrained_or_selected_calls, node)
        push!(state.value_changed, node)
        push!(state.discard_calls, node)
    end
    if any(input_node in state.value_changed for input_node in node.inputs)
        push!(state.input_changed, node)
        push!(state.value_changed, node)
        push!(state.discard_calls, node)
    end
end

#########################
# backward marking pass #
#########################

struct BackwardPassState 
    marked::Set{StaticIRNode}
end

function BackwardPassState()
    BackwardPassState(Set{StaticIRNode}())
end

function process_backward!(::ForwardPassState, ::BackwardPassState, ::ArgumentNode) end

function process_backward!(::ForwardPassState, ::BackwardPassState, node) end

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
    if node in fwd.input_changed || node in fwd.constrained_or_selected_choices
        for input_node in node.inputs
            push!(back.marked, input_node)
        end
    end
end

function process_backward!(fwd::ForwardPassState, back::BackwardPassState,
                           node::GenerativeFunctionCallNode)
    if node in fwd.input_changed || node in fwd.constrained_or_selected_calls
        for input_node in node.inputs
            push!(back.marked, input_node)
        end
    end
end

########################
# code generation pass #
########################

function arg_values_and_diffs(input_nodes)
    arg_values = map((node) -> Expr(:call, QuoteNode(strip_diff), node.name), input_nodes)
    arg_diffs = map((node) -> Expr(:call, QuoteNode(get_diff), node.name), input_nodes)
    (arg_values, arg_diffs)
end

function process_codegen!(stmts, ::ForwardPassState, ::BackwardPassState,
                          node::ArgumentNode, ::AbstractUpdateMode)
    push!(stmts, :($(get_value_fieldname(node)) = $(node.name)))
end

function process_codegen!(stmts, fwd::ForwardPassState, back::BackwardPassState,
                         node::JuliaNode, ::AbstractUpdateMode)
    if node in back.marked
        arg_values, arg_diffs = arg_values_and_diffs(node.inputs)
        args = map((v, d) -> Expr(:call, QuoteNode(Diffed), v, d), arg_values, arg_diffs)
        push!(stmts, :($(node.name) = $(QuoteNode(node.fn))($(args...))))
    end
end

function process_codegen!(stmts, fwd::ForwardPassState, back::BackwardPassState,
                          node::RandomChoiceNode, ::Union{UpdateMode,ExtendMode})
    arg_values, _ = arg_values_and_diffs(node.inputs)
    new_logpdf = gensym("new_logpdf")
    addr = QuoteNode(node.addr)
    dist = QuoteNode(node.dist)
    if node in fwd.constrained_or_selected_choices || node in fwd.input_changed
        if node in fwd.constrained_or_selected_choices
            push!(stmts, :($(node.name) = Diffed(static_get_value(constraints, Val($addr)), UnknownChange())))
            push!(stmts, :($(choice_discard_var(node)) = trace.$(get_value_fieldname(node))))
        else
            push!(stmts, :($(node.name) = Diffed(trace.$(get_value_fieldname(node)), NoChange())))
        end
        push!(stmts, :($new_logpdf = logpdf($dist, $(Expr(:call, QuoteNode(strip_diff), node.name)), $(arg_values...))))
        push!(stmts, :($weight += $new_logpdf - trace.$(get_score_fieldname(node))))
        push!(stmts, :($total_score_fieldname += $new_logpdf - trace.$(get_score_fieldname(node))))
        push!(stmts, :($(get_score_fieldname(node)) = $new_logpdf))
    else
        push!(stmts, :($(node.name) = Diffed(trace.$(get_value_fieldname(node)), NoChange())))
        push!(stmts, :($(get_score_fieldname(node)) = trace.$(get_score_fieldname(node))))
    end
    push!(stmts, :($(get_value_fieldname(node)) = $(node.name)))
end

function process_codegen!(stmts, fwd::ForwardPassState, back::BackwardPassState,
                          node::RandomChoiceNode, ::RegenerateMode)
    arg_values, _ = arg_values_and_diffs(node.inputs)
    new_logpdf = gensym("new_logpdf")
    addr = QuoteNode(node.addr)
    dist = QuoteNode(node.dist)
    if node in fwd.constrained_or_selected_choices || node in fwd.input_changed
        if node in fwd.constrained_or_selected_choices
            # the choice was selected, it does not contribute to the weight
            push!(stmts, :($(node.name) = Diffed(random($dist, $(arg_values...)), UnknownChange())))
            push!(stmts, :($new_logpdf = logpdf($dist, $(Expr(:call, QuoteNode(strip_diff), node.name)), $(arg_values...))))
        else
            # the choice was not selected, and the input to the choice changed
            # it does contribute to the weight
            push!(stmts, :($(node.name) = Diffed(trace.$(get_value_fieldname(node)), NoChange())))
            push!(stmts, :($new_logpdf = logpdf($dist, $(Expr(:call, QuoteNode(strip_diff), node.name)), $(arg_values...))))
            push!(stmts, :($weight += $new_logpdf - trace.$(get_score_fieldname(node))))
        end
        push!(stmts, :($total_score_fieldname += $new_logpdf - trace.$(get_score_fieldname(node))))
        push!(stmts, :($(get_score_fieldname(node)) = $new_logpdf))
    else
        push!(stmts, :($(node.name) = Diffed(trace.$(get_value_fieldname(node)), NoChange())))
        push!(stmts, :($(get_score_fieldname(node)) = trace.$(get_score_fieldname(node))))
    end
    push!(stmts, :($(get_value_fieldname(node)) = $(node.name)))
end

function process_codegen!(stmts, fwd::ForwardPassState, back::BackwardPassState,
                          node::GenerativeFunctionCallNode, mode::AbstractUpdateMode)
    arg_values, arg_diffs = arg_values_and_diffs(node.inputs)
    addr = QuoteNode(node.addr)
    gen_fn = QuoteNode(node.generative_function)
    subtrace = get_subtrace_fieldname(node)
    prev_subtrace = :(trace.$subtrace)
    call_weight = gensym("call_weight")
    call_constraints = gensym("call_constraints")
    if node in fwd.constrained_or_selected_calls || node in fwd.input_changed
        if node in fwd.constrained_or_selected_calls
            push!(stmts, :($call_constraints = static_get_submap(constraints, Val($addr))))
        else
            push!(stmts, :($call_constraints = EmptyChoiceMap()))
        end
        push!(stmts, :(($subtrace, $call_weight, $(calldiff_var(node)), $(call_discard_var(node))) = 
            update($prev_subtrace, $(Expr(:tuple, arg_values...)), $(Expr(:tuple, arg_diffs...)), $call_constraints)))
        push!(stmts, :($weight += $call_weight))
        push!(stmts, :($total_score_fieldname += get_score($subtrace) - get_score($prev_subtrace)))
        push!(stmts, :($total_noise_fieldname += project($subtrace, EmptyAddressSet()) - project($prev_subtrace, EmptyAddressSet())))
        push!(stmts, :(if !isempty(get_choices($subtrace)) && isempty(get_choices($prev_subtrace))
                            $num_nonempty_fieldname += 1 end))
        push!(stmts, :(if isempty(get_choices($subtrace)) && !isempty(get_choices($prev_subtrace))
                            $num_nonempty_fieldname -= 1 end))
        push!(stmts, :($(node.name) = Diffed(get_retval($subtrace), $(calldiff_var(node)))))
    else
        push!(stmts, :($subtrace = $prev_subtrace))
        push!(stmts, :($(node.name) = Diffed(get_retval($subtrace), NoChange())))
    end
end

function process_codegen!(stmts, fwd::ForwardPassState, back::BackwardPassState,
                          node::GenerativeFunctionCallNode, ::RegenerateMode)
    arg_values, arg_diffs = arg_values_and_diffs(node.inputs)
    addr = QuoteNode(node.addr)
    gen_fn = QuoteNode(node.generative_function)
    subtrace = get_subtrace_fieldname(node)
    prev_subtrace = :(trace.$subtrace)
    call_weight = gensym("call_weight")
    call_subselection = gensym("call_subselection")
    if node in fwd.constrained_or_selected_calls || node in fwd.input_changed
        if node in fwd.constrained_or_selected_calls
            push!(stmts, :($call_subselection = Gen.static_get_internal_node(selection, Val($addr))))
        else
            push!(stmts, :($call_subselection = EmptyAddressSet()))
        end
        push!(stmts, :(($subtrace, $call_weight, $(calldiff_var(node))) = 
            regenerate($prev_subtrace, $(Expr(:tuple, arg_values...)), $(Expr(:tuple, arg_diffs...)), $call_subselection)))
        push!(stmts, :($weight += $call_weight))
        push!(stmts, :($total_score_fieldname += get_score($subtrace) - get_score($prev_subtrace)))
        push!(stmts, :($total_noise_fieldname += project($subtrace, EmptyAddressSet()) - project($prev_subtrace, EmptyAddressSet())))
        push!(stmts, :(if !isempty(get_choices($subtrace)) && !isempty(get_choices($prev_subtrace))
                            $num_nonempty_fieldname += 1 end))
        push!(stmts, :(if isempty(get_choices($subtrace)) && !isempty(get_choices($prev_subtrace))
                            $num_nonempty_fieldname -= 1 end))
        push!(stmts, :($(node.name) = Diffed(get_retval($subtrace), $(calldiff_var(node)))))
    else
        push!(stmts, :($subtrace = $prev_subtrace))
        push!(stmts, :($(node.name) = Diffed(get_retval($subtrace), NoChange())))
    end
end

function process_codegen!(stmts, fwd::ForwardPassState, back::BackwardPassState,
                          node::GenerativeFunctionCallNode, ::ExtendMode)
    arg_values, arg_diffs = arg_values_and_diffs(node.inputs)
    addr = QuoteNode(node.addr)
    gen_fn = QuoteNode(node.generative_function)
    subtrace = get_subtrace_fieldname(node)
    prev_subtrace = :(trace.$subtrace)
    call_weight = gensym("call_weight")
    call_constraints = gensym("call_constraints")
    if node in fwd.constrained_or_selected_calls || node in fwd.input_changed
        if node in fwd.constrained_or_selected_calls
            push!(stmts, :($call_constraints = static_get_submap(constraints, Val($addr))))
        else
            push!(stmts, :($call_constraints = EmptyChoiceMap()))
        end
        push!(stmts, :(($subtrace, $call_weight, $(calldiff_var(node))) = 
            extend($prev_subtrace, $(Expr(:tuple, arg_values...)), $(Expr(:tuple, arg_diffs...)), $call_constraints)
        ))
        push!(stmts, :($weight += $call_weight))
        push!(stmts, :($total_score_fieldname += $call_weight))
        push!(stmts, :($total_noise_fieldname += project($subtrace, EmptyAddressSet()) - project($prev_subtrace, EmptyAddressSet())))
        push!(stmts, :(if !isempty(get_choices($subtrace)) && !isempty(get_choices($prev_subtrace))
                            $num_nonempty_fieldname += 1 end))
        push!(stmts, :(if isempty(get_choices($subtrace)) && !isempty(get_choices($prev_subtrace))
                            $num_nonempty_fieldname -= 1 end))
        push!(stmts, :($(node.name) = Diffed(get_retval($subtrace), $(calldiff_var(node)))))
    else
        push!(stmts, :($subtrace = $prev_subtrace))
        push!(stmts, :($(node.name) = Diffed(get_retval($subtrace), NoChange())))
    end
end

function initialize_score_weight_num_nonempty!(stmts::Vector{Expr})
    push!(stmts, :($total_score_fieldname = trace.$total_score_fieldname))
    push!(stmts, :($total_noise_fieldname = trace.$total_noise_fieldname))
    push!(stmts, :($weight = 0.))
    push!(stmts, :($num_nonempty_fieldname = trace.$num_nonempty_fieldname))
end

function unpack_arguments!(stmts::Vector{Expr}, arg_nodes::Vector{ArgumentNode})
    arg_names = Symbol[arg_node.name for arg_node in arg_nodes]
    push!(stmts, :($(Expr(:tuple, arg_names...)) = args))
end

function generate_return_value!(stmts::Vector{Expr}, return_node::StaticIRNode)
    push!(stmts, :($return_value_fieldname = $(Expr(:call, QuoteNode(strip_diff), return_node.name))))
    push!(stmts, :($retdiff = $(Expr(:call, QuoteNode(get_diff), return_node.name))))
end

function generate_new_trace!(stmts::Vector{Expr}, trace_type::Type)
    constructor_args = map((name) -> Expr(:call, QuoteNode(strip_diff), name), fieldnames(trace_type))
    push!(stmts, :($trace = $(QuoteNode(trace_type))($(constructor_args...))))
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
    expr = :(StaticChoiceMap(
            NamedTuple{($(leaf_keys...),)}(($(leaf_nodes...),)),
            NamedTuple{($(internal_keys...),)}(($(internal_nodes...),))))
    push!(stmts, :($discard = $expr))
end

#######################
# generated functions #
#######################

function codegen_update(trace_type::Type{T}, args_type::Type, argdiffs_type::Type,
                        constraints_type::Type) where {T<:StaticIRTrace}
    gen_fn_type = get_gen_fn_type(trace_type)
    schema = get_address_schema(constraints_type)

    # convert the constraints to a static assignment if it is not already one
    if !(isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema))
        return quote update(trace, args, argdiffs, StaticChoiceMap(constraints)) end
    end

    ir = get_ir(gen_fn_type)

    # forward marking pass
    fwd_state = ForwardPassState()
    forward_pass_argdiff!(fwd_state, ir.arg_nodes, argdiffs_type)
    for node in ir.nodes
        process_forward!(schema, fwd_state, node)
    end

    # backward marking pass
    bwd_state = BackwardPassState()
    push!(bwd_state.marked, ir.return_node)
    for node in reverse(ir.nodes)
        process_backward!(fwd_state, bwd_state, node)
    end

    # forward code generation pass
    stmts = Expr[]
    initialize_score_weight_num_nonempty!(stmts)
    unpack_arguments!(stmts, ir.arg_nodes)
    for node in ir.nodes
        process_codegen!(stmts, fwd_state, bwd_state, node, UpdateMode())
    end
    generate_return_value!(stmts, ir.return_node)
    generate_new_trace!(stmts, trace_type)
    generate_discard!(stmts, fwd_state.constrained_or_selected_choices, fwd_state.discard_calls)

    # return trace and weight and discard and retdiff
    push!(stmts, :(return ($trace, $weight, $retdiff, $discard)))

    Expr(:block, stmts...)
end

function codegen_regenerate(trace_type::Type{T}, args_type::Type, argdiffs_type::Type,
                            selection_type::Type) where {T<:StaticIRTrace}
    gen_fn_type = get_gen_fn_type(trace_type)
    schema = get_address_schema(selection_type)

    # convert the constraints to a static assignment if it is not already one
    if !(isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema))
        return quote regenerate(trace, args, argdiffs, StaticAddressSet(selection)) end
    end

    ir = get_ir(gen_fn_type)

    # forward marking pass
    fwd_state = ForwardPassState()
    forward_pass_argdiff!(fwd_state, ir.arg_nodes, argdiffs_type)
    for node in ir.nodes
        process_forward!(schema, fwd_state, node)
    end

    # backward marking pass
    bwd_state = BackwardPassState()
    push!(bwd_state.marked, ir.return_node)
    for node in reverse(ir.nodes)
        process_backward!(fwd_state, bwd_state, node)
    end

    # forward code generation pass
    stmts = Expr[]
    initialize_score_weight_num_nonempty!(stmts)
    unpack_arguments!(stmts, ir.arg_nodes)
    for node in ir.nodes
        process_codegen!(stmts, fwd_state, bwd_state, node, RegenerateMode())
    end
    generate_return_value!(stmts, ir.return_node)
    generate_new_trace!(stmts, trace_type)

    # return trace and weight and retdiff
    push!(stmts, :(return ($trace, $weight, $retdiff)))

    Expr(:block, stmts...)
end

function codegen_extend(trace_type::Type{T}, args_type::Type, argdiffs_type::Type,
                        constraints_type::Type) where {T<:StaticIRTrace}
    gen_fn_type = get_gen_fn_type(trace_type)
    schema = get_address_schema(constraints_type)

    # convert the constraints to a static assignment if it is not already one
    if !(isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema))
        return quote extend(trace, args, argdiffs, StaticChoiceMap(constraints)) end
    end

    ir = get_ir(gen_fn_type)

    # forward marking pass
    fwd_state = ForwardPassState()
    forward_pass_argdiff!(fwd_state, ir.arg_nodes, argdiffs_type)
    for node in ir.nodes
        process_forward!(schema, fwd_state, node)
    end

    # backward marking pass
    bwd_state = BackwardPassState()
    push!(bwd_state.marked, ir.return_node)
    for node in reverse(ir.nodes)
        process_backward!(fwd_state, bwd_state, node)
    end

    # forward code generation pass
    stmts = Expr[]
    initialize_score_weight_num_nonempty!(stmts)
    unpack_arguments!(stmts, ir.arg_nodes)
    for node in ir.nodes
        process_codegen!(stmts, fwd_state, bwd_state, node, ExtendMode())
    end
    generate_return_value!(stmts, ir.return_node)
    generate_new_trace!(stmts, trace_type)

    # return trace and weight and retdiff
    push!(stmts, :(return ($trace, $weight, $retdiff)))

    Expr(:block, stmts...)
end

push!(Gen.generated_functions, quote
@generated function Gen.update(trace::T, args::Tuple, argdiffs::Tuple,
                               constraints::ChoiceMap) where {T<:StaticIRTrace}
    Gen.codegen_update(trace, args, argdiffs, constraints)
end
end)

push!(Gen.generated_functions, quote
@generated function Gen.regenerate(trace::T, args::Tuple, argdiffs::Tuple,
                                   selection::AddressSet) where {T<:StaticIRTrace}
    Gen.codegen_regenerate(trace, args, argdiffs, selection)
end
end)

push!(Gen.generated_functions, quote
@generated function Gen.extend(trace::T, args::Tuple, argdiffs::Tuple,
                               constraints::ChoiceMap) where {T<:StaticIRTrace}
    Gen.codegen_extend(trace, args, argdiffs, constraints)
end
end)
