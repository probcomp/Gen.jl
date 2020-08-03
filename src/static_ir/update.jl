const retdiff = gensym("retdiff")
const discard = gensym("discard")

const calldiff_prefix = gensym("calldiff")
calldiff_var(node::GenerativeFunctionCallNode) = Symbol("$(calldiff_prefix)_$(node.addr)")

const choice_discard_prefix = gensym("choice_discard")

const call_discard_prefix = gensym("call_discard")
call_discard_var(node::GenerativeFunctionCallNode) = Symbol("$(call_discard_prefix)_$(node.addr)")

########################
# forward marking pass #
########################

struct ForwardPassState
    input_changed::Set{GenerativeFunctionCallNode}
    value_changed::Set{StaticIRNode}
    constrained_or_selected_calls::Set{GenerativeFunctionCallNode}
    discard_calls::Set{GenerativeFunctionCallNode}
end

function ForwardPassState()
    input_changed = Set{GenerativeFunctionCallNode}()
    value_changed = Set{StaticIRNode}()
    constrained_or_selected_calls = Set{GenerativeFunctionCallNode}()
    discard_calls = Set{GenerativeFunctionCallNode}()
    ForwardPassState(input_changed, value_changed, constrained_or_selected_calls, discard_calls)
end

function forward_pass_argdiff!(state::ForwardPassState,
                               arg_nodes::Vector{ArgumentNode},
                               argdiffs_type::Type)
    for (node, diff_type) in zip(arg_nodes, argdiffs_type.parameters)
        if diff_type != NoChange
            push!(state.value_changed, node)
        end
    end
end

function process_forward!(::Type{<:UpdateSpec}, ::Type{<:Selection}, ::ForwardPassState, ::TrainableParameterNode) end

function process_forward!(::Type{<:UpdateSpec}, ::Type{<:Selection}, ::ForwardPassState, node::ArgumentNode) end

function process_forward!(::Type{<:UpdateSpec}, ::Type{<:Selection}, state::ForwardPassState, node::JuliaNode)
    if any(input_node in state.value_changed for input_node in node.inputs)
        push!(state.value_changed, node)
    end
end

function cannot_statically_guarantee_nochange_retdiff(spec_type, externally_constrained_addr_type, node, state)
    trace_type = get_trace_type(node.generative_function)
    argdiff_types = map(input_node -> input_node in state.value_changed ? UnknownChange : NoChange, node.inputs)
    argdiff_type = Tuple{argdiff_types...}
    # TODO: can we know the arg type statically?

    # TODO: is there a way to do this using `get_subtree` with constant propagation?
    # if so, we might be able to avoid always casting to static address trees
    subspec_type = Core.Compiler.return_type(static_get_subtree, Tuple{spec_type, Val{node.addr}})
    subext_const_addr_type = Core.Compiler.return_type(static_get_subtree, Tuple{externally_constrained_addr_type, Val{node.addr}})
    
    update_rettype = Core.Compiler.return_type(
        Gen.update,
        Tuple{trace_type, Tuple, argdiff_type, subspec_type, subext_const_addr_type}
    )
    has_static_retdiff = update_rettype <: Tuple && update_rettype != Union{} && length(update_rettype.parameters) >= 3
    guaranteed_returns_nochange = has_static_retdiff && update_rettype.parameters[3] == NoChange

    return !guaranteed_returns_nochange
end

function process_forward!(spec_type::Type{<:UpdateSpec}, externally_constrained_addrs_type::Type{<:Selection},
                          state::ForwardPassState,
                          node::GenerativeFunctionCallNode)
    schema = get_address_schema(spec_type)
    will_run_update = false
    @assert isa(schema, StaticSchema)
    if isa(schema, AllAddressSchema) || (!isa(schema, EmptyAddressSchema) && node.addr in keys(schema))
        push!(state.constrained_or_selected_calls, node)
        will_run_update = true
    end
    if any(input_node in state.value_changed for input_node in node.inputs)
        push!(state.input_changed, node)
        will_run_update = true
    end
    if will_run_update
        push!(state.discard_calls, node)
        if cannot_statically_guarantee_nochange_retdiff(spec_type, externally_constrained_addrs_type, node, state)
            push!(state.value_changed, node)
        end
    end
end

#########################
# backward marking pass #
#########################

# this pass is used to determine which JuliaNodes need to be re-run (their
# return value is not currently cached in the trace)

struct BackwardPassState
    marked::Set{StaticIRNode}
end

function BackwardPassState()
    BackwardPassState(Set{StaticIRNode}())
end

function process_backward!(::ForwardPassState, ::BackwardPassState, ::TrainableParameterNode, options) end

function process_backward!(::ForwardPassState, ::BackwardPassState, ::ArgumentNode, options) end

function process_backward!(fwd::ForwardPassState, back::BackwardPassState,
                           node::JuliaNode, options)
    # if the node needs to be re-run which then we need the value of its inputs
    if ((options.cache_julia_nodes && node in fwd.value_changed) ||
        (!options.cache_julia_nodes && node in back.marked))
        for input_node in node.inputs
            push!(back.marked, input_node)
        end
    end
end

function process_backward!(fwd::ForwardPassState, back::BackwardPassState,
                           node::GenerativeFunctionCallNode, options)
    if node in fwd.input_changed || node in fwd.constrained_or_selected_calls
        for input_node in node.inputs
            push!(back.marked, input_node)
        end
    end
end

########################
# code generation pass #
########################

function arg_values_and_diffs_from_tracked_diffs(input_nodes)
    arg_values = map((node) -> Expr(:call, (GlobalRef(Gen, :strip_diff)), node.name), input_nodes)
    arg_diffs = map((node) -> Expr(:call, (GlobalRef(Gen, :get_diff)), node.name), input_nodes)
    (arg_values, arg_diffs)
end

function process_codegen!(stmts, ::ForwardPassState, back::BackwardPassState,
                          node::TrainableParameterNode, options)
    if node in back.marked
        push!(stmts, :($(node.name) = $(QuoteNode(get_param))($(QuoteNode(get_gen_fn))(trace), $(QuoteNode(node.name)))))
    end
end

function process_codegen!(stmts, ::ForwardPassState, ::BackwardPassState,
                          node::ArgumentNode, options)
    if options.track_diffs
        push!(stmts, :($(get_value_fieldname(node)) = $(GlobalRef(Gen, :strip_diff))($(node.name))))
    else
        push!(stmts, :($(get_value_fieldname(node)) = $(node.name)))
    end
end

function process_codegen!(stmts, fwd::ForwardPassState, back::BackwardPassState,
                         node::JuliaNode, options)
    if options.track_diffs
        arg_values, arg_diffs = arg_values_and_diffs_from_tracked_diffs(node.inputs)
        args = map((v, d) -> Expr(:call, (GlobalRef(Gen, :Diffed)), v, d), arg_values, arg_diffs)
        if !options.cache_julia_nodes && (node in back.marked)
            push!(stmts, :($(node.name) = $(QuoteNode(node.fn))($(args...))))
        elseif node in fwd.value_changed
            push!(stmts, (quote
               if !($(Expr(:call, GlobalRef(Gen, :all_nochange), Expr(:tuple, arg_diffs...))))
                    $(node.name) = $(QuoteNode(node.fn))($(args...))
                else
                    $(node.name) = $(GlobalRef(Gen, :Diffed))(trace.$(get_value_fieldname(node)), $(GlobalRef(Gen, :NoChange))())
                end
            end).args[2])
        else
            push!(stmts, :($(node.name) = $(GlobalRef(Gen, :Diffed))(trace.$(get_value_fieldname(node)), $(GlobalRef(Gen, :NoChange))())))
        end
        if options.cache_julia_nodes
            push!(stmts, :($(get_value_fieldname(node)) = $(GlobalRef(Gen, :strip_diff))($(node.name))))
        end
    else
        if (!options.cache_julia_nodes && node in back.marked) || (node in fwd.value_changed)
            arg_values = map((n) -> n.name, node.inputs)
            push!(stmts, :($(node.name) = $(QuoteNode(node.fn))($(arg_values...))))
        elseif options.cache_julia_nodes
            push!(stmts, :($(node.name) = trace.$(get_value_fieldname(node))))
        end
        if options.cache_julia_nodes
            push!(stmts, :($(get_value_fieldname(node)) = $(node.name)))
        end
    end
end

function process_codegen!(stmts, fwd::ForwardPassState, back::BackwardPassState,
                          node::GenerativeFunctionCallNode, options)
    if options.track_diffs
        arg_values, arg_diffs = arg_values_and_diffs_from_tracked_diffs(node.inputs)
    else
        arg_values = map((n) -> n.name, node.inputs)
        arg_diffs = map((n) -> QuoteNode(n in fwd.value_changed ? UnknownChange() : NoChange()), node.inputs)
    end
    addr = QuoteNode(node.addr)
    gen_fn = QuoteNode(node.generative_function)
    subtrace = get_subtrace_fieldname(node)
    prev_subtrace = :(trace.$subtrace)
    call_weight = gensym("call_weight")
    call_spec = gensym("call_spec")
    ext_const_addrs = gensym("ext_const_addrs")
    if node in fwd.constrained_or_selected_calls || node in fwd.input_changed
        if node in fwd.constrained_or_selected_calls
            push!(stmts, :($call_spec = $(GlobalRef(Gen, :static_get_subtree))(spec, Val($addr))))
        else
            push!(stmts, :($call_spec = $(GlobalRef(Gen, :EmptyAddressTree))()))
        end
        push!(stmts, :($ext_const_addrs = $(GlobalRef(Gen, :get_subtree))(externally_constrained_addrs, $addr)))
        push!(stmts, :(($subtrace, $call_weight, $(calldiff_var(node)), $(call_discard_var(node))) =
            $(GlobalRef(Gen, :update))($prev_subtrace, $(Expr(:tuple, arg_values...)), $(Expr(:tuple, arg_diffs...)), $call_spec, $ext_const_addrs)))
        push!(stmts, :($weight += $call_weight))
        push!(stmts, :($total_score_fieldname += $(GlobalRef(Gen, :get_score))($subtrace) - $(GlobalRef(Gen, :get_score))($prev_subtrace)))
        push!(stmts, :($total_noise_fieldname += $(GlobalRef(Gen, :project))($subtrace, $(GlobalRef(Gen, :EmptySelection))()) - $(GlobalRef(Gen, :project))($prev_subtrace, $(GlobalRef(Gen, :EmptySelection))())))
        push!(stmts, :(if !$(GlobalRef(Gen, :isempty))($(GlobalRef(Gen, :get_choices))($subtrace)) && $(GlobalRef(Gen, :isempty))($(GlobalRef(Gen, :get_choices))($prev_subtrace))
                            $num_nonempty_fieldname += 1 end))
        push!(stmts, :(if $(GlobalRef(Gen, :isempty))($(GlobalRef(Gen, :get_choices))($subtrace)) && !$(GlobalRef(Gen, :isempty))($(GlobalRef(Gen, :get_choices))($prev_subtrace))
                            $num_nonempty_fieldname -= 1 end))
        if options.track_diffs
            push!(stmts, :($(node.name) = $(GlobalRef(Gen, :Diffed))($(GlobalRef(Gen, :get_retval))($subtrace), $(calldiff_var(node)))))
        else
            push!(stmts, :($(node.name) = $(GlobalRef(Gen, :get_retval))($subtrace)))
        end
    else
        push!(stmts, :($subtrace = $prev_subtrace))
        if options.track_diffs
            push!(stmts, :($(node.name) = $(GlobalRef(Gen, :Diffed))($(GlobalRef(Gen, :get_retval))($subtrace), $(GlobalRef(Gen, :NoChange))())))
        else
            push!(stmts, :($(node.name) = $(GlobalRef(Gen, :get_retval))($subtrace)))
        end
    end
end

function initialize_score_weight_num_nonempty!(stmts::Vector{Expr})
    push!(stmts, :($total_score_fieldname = trace.$total_score_fieldname))
    push!(stmts, :($total_noise_fieldname = trace.$total_noise_fieldname))
    push!(stmts, :($weight = 0.))
    push!(stmts, :($num_nonempty_fieldname = trace.$num_nonempty_fieldname))
end

function unpack_arguments!(stmts::Vector{Expr}, arg_nodes::Vector{ArgumentNode}, options)
    if options.track_diffs
        arg_names = Symbol[arg_node.name for arg_node in arg_nodes]
        push!(stmts, :($(Expr(:tuple, arg_names...)) = $(QuoteNode(map))($(GlobalRef(Gen, :Diffed)), args, argdiffs)))
    else
        arg_names = Symbol[arg_node.name for arg_node in arg_nodes]
        push!(stmts, :($(Expr(:tuple, arg_names...)) = args))
    end
end

function generate_return_value!(stmts::Vector{Expr}, fwd::ForwardPassState, return_node::StaticIRNode, options)
    if options.track_diffs
        push!(stmts, :($return_value_fieldname = $(GlobalRef(Gen, :strip_diff))($(return_node.name))))
        push!(stmts, :($retdiff = $(GlobalRef(Gen, :get_diff))($(return_node.name))))
    else
        push!(stmts, :($return_value_fieldname = $(return_node.name)))
        push!(stmts, :($retdiff = $(QuoteNode(return_node in fwd.value_changed ? UnknownChange() : NoChange()))))
    end
end

function generate_new_trace!(stmts::Vector{Expr}, trace_type::Type, options)
    if options.track_diffs
        # note that the generative function is the last field
        constructor_args = map((name) -> Expr(:call, QuoteNode(strip_diff), name),
                            fieldnames(trace_type)[1:end-1])
        push!(stmts, :($trace = $(QuoteNode(trace_type))($(constructor_args...),
                        $(Expr(:(.), :trace, QuoteNode(static_ir_gen_fn_ref))))))
    else
        push!(stmts, :($static_ir_gen_fn_ref = $(Expr(:(.), :trace, QuoteNode(static_ir_gen_fn_ref)))))
        push!(stmts, :($trace = $(QuoteNode(trace_type))($(fieldnames(trace_type)...))))
    end
end

function generate_discard!(stmts::Vector{Expr}, discard_calls::Set{GenerativeFunctionCallNode})
    discard_nodes = Dict{Symbol,Symbol}()
    for node in discard_calls
        discard_nodes[node.addr] = call_discard_var(node)
    end

    if length(discard_nodes) > 0
        (keys, nodes) = collect(zip(discard_nodes...))
    else
        (keys, nodes) = ((), ())
    end
    keys = map((key::Symbol) -> QuoteNode(key), keys)
    expr = quote $(QuoteNode(StaticChoiceMap))(
            $(QuoteNode(NamedTuple)){($(keys...),)}(($(nodes...),))) end
    push!(stmts, :($discard = $expr))
end

#######################
# generated functions #
#######################

function codegen_update(trace_type::Type{T}, args_type::Type, argdiffs_type::Type,
                        spec_type::Type, externally_constrained_addrs_type::Type) where {T<:StaticIRTrace}
    gen_fn_type = get_gen_fn_type(trace_type)
    spec_schema = get_address_schema(spec_type)
    ext_const_addrs_schema = get_address_schema(externally_constrained_addrs_type)

    spec_is_static = isa(spec_schema, StaticSchema)
    ext_const_addrs_is_static = isa(ext_const_addrs_schema, StaticSchema)

    # convert the spec and ext_const_addrs to static if they are not already
    if !(spec_is_static && ext_const_addrs_is_static)
        return quote $(GlobalRef(Gen, :update))(trace, args, argdiffs, $(QuoteNode(StaticAddressTree))(spec), $(QuoteNode(StaticAddressTree))(externally_constrained_addrs)) end
    end

    ir = get_ir(gen_fn_type)
    options = get_options(gen_fn_type)

    # forward marking pass
    fwd_state = ForwardPassState()
    forward_pass_argdiff!(fwd_state, ir.arg_nodes, argdiffs_type)
    for node in ir.nodes
        process_forward!(spec_type, externally_constrained_addrs_type, fwd_state, node)
    end

    # backward marking pass
    # TODO computes which values do we need to recompute - if we cache all Julia values, then we don't need to recompute anything.
    # TODO: we still need to extract only:
    # - trainable parameters nodes that we need
    # - julia node values that we need
    bwd_state = BackwardPassState()
    push!(bwd_state.marked, ir.return_node)
    for node in reverse(ir.nodes)
        process_backward!(fwd_state, bwd_state, node, options)
    end

    # forward code generation pass
    stmts = Expr[]
    initialize_score_weight_num_nonempty!(stmts)
    unpack_arguments!(stmts, ir.arg_nodes, options)
    for node in ir.nodes
        process_codegen!(stmts, fwd_state, bwd_state, node, options)
    end
    generate_return_value!(stmts, fwd_state, ir.return_node, options)
    generate_new_trace!(stmts, trace_type, options)
    generate_discard!(stmts, fwd_state.discard_calls)

    # return trace and weight and discard and retdiff
    push!(stmts, :(return ($trace, $weight, $retdiff, $discard)))

    Expr(:block, stmts...)
end

let T = gensym()
    push!(generated_functions, quote
    @generated function $(GlobalRef(Gen, :update))(trace::$T, args::Tuple, argdiffs::Tuple,
                                   spec::$(QuoteNode(UpdateSpec)), externally_constrained_addrs::$(QuoteNode(Selection))) where {$T<:$(QuoteNode(StaticIRTrace))}
        $(QuoteNode(codegen_update))(trace, args, argdiffs, spec, externally_constrained_addrs)
    end
    end)
end
