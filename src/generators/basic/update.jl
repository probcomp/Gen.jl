const bb_score = gensym("score")
const bb_weight = gensym("weight")
const bb_new_trace = gensym("trace")

struct BBUpdateState
    marked::Set{ValueNode}
    stmts::Vector{Expr}
    schema::Union{StaticAddressSchema,EmptyAddressSchema}
    addr_visited::Set{Symbol}
    discard_leaf_nodes::Dict{Symbol, Symbol}
    discard_internal_nodes::Dict{Symbol, Symbol}
end

function BBUpdateState(stmts::Vector{Expr}, ir::BasicBlockIR, schema::Union{StaticAddressSchema,EmptyAddressSchema}, args_change_type)
    addr_visited = Set{Symbol}()
    marked = Set{ValueNode}()
    mark_arguments!(marked, ir, args_change_type)
    mark_input_change_nodes!(marked, ir)
    discard_leaf_node = Dict{Symbol,Symbol}()
    discard_internal_node = Dict{Symbol,Symbol}()
    BBUpdateState(marked, stmts, schema, addr_visited, discard_leaf_node, discard_internal_node)
end

struct BBFixUpdateState 
    marked::Set{ValueNode}
    stmts::Vector{Expr}
    schema::Union{StaticAddressSchema,EmptyAddressSchema}
    addr_visited::Set{Symbol}
    discard_leaf_nodes::Dict{Symbol, Symbol}
    discard_internal_nodes::Dict{Symbol, Symbol}
end

function BBFixUpdateState(stmts::Vector{Expr}, ir::BasicBlockIR, schema::Union{StaticAddressSchema,EmptyAddressSchema}, args_change_type)
    addr_visited = Set{Symbol}()
    marked = Set{ValueNode}()
    mark_arguments!(marked, ir, args_change_type)
    mark_input_change_nodes!(marked, ir)
    discard_leaf_node = Dict{Symbol,Symbol}()
    discard_internal_node = Dict{Symbol,Symbol}()
    BBFixUpdateState(marked, stmts, schema, addr_visited, discard_leaf_node, discard_internal_node)
end

struct BBExtendState
    marked::Set{ValueNode}
    stmts::Vector{Expr}
    schema::Union{StaticAddressSchema,EmptyAddressSchema}
    addr_visited::Set{Symbol}
end

function BBExtendState(stmts::Vector{Expr}, ir::BasicBlockIR, schema::Union{StaticAddressSchema,EmptyAddressSchema}, args_change_type)
    addr_visited = Set{Symbol}()
    marked = Set{ValueNode}()
    mark_arguments!(marked, ir, args_change_type)
    mark_input_change_nodes!(marked, ir)
    BBExtendState(marked, stmts, schema, addr_visited)
end

function mark_input_change_nodes!(marked::Set{ValueNode}, ir::BasicBlockIR)
    for node in ir.generator_input_change_nodes
        # for now, mark every input change node to a generator
        # TODO we should only mark the node if the corresponding generator is
        # either constrained or marked, however, this requires two passes.
        # postponed for simplicity.
        push!(marked, node)
    end
end

function mark_arguments!(marked::Set{ValueNode}, ir::BasicBlockIR, args_change::Type{UnknownArgDiff})
    for arg_node in ir.arg_nodes
        push!(marked, arg_node)
    end
end

function mark_arguments!(marked::Set{ValueNode}, ir::BasicBlockIR, args_change::Type{NoArgDiff}) end

function mark_arguments!(marked::Set{ValueNode}, ir::BasicBlockIR, args_change::Type{T}) where {T <: MaskedArgChange}
    mask = args_change.parameters[1].parameters
    for (arg_node, maybe_changed_val) in zip(ir.arg_nodes, mask)
        if maybe_changed_val.parameters[1]
            push!(marked, arg_node)
        end
    end
end

function process!(ir::BasicBlockIR, state::Union{BBUpdateState,BBFixUpdateState,BBExtendState}, node::JuliaNode)
    
    # if any input nodes are marked, mark the output node
    if any([input in state.marked for input in node.input_nodes])
        push!(state.marked, node.output)
    end

    # set the value in the new trace based on other values in the new trace
    (typ, trace_field) = get_value_info(node)
    if node.output in state.marked
        push!(state.stmts, quote
            $bb_new_trace.$trace_field = $(expr_read_from_trace(node, bb_new_trace))
        end)
    end
end

function process!(ir::BasicBlockIR, state::Union{BBUpdateState,BBFixUpdateState,BBExtendState}, node::ArgsChangeNode)
    # always mark
    push!(state.marked, node.output)

    # set the value in the new trace (in the future, for performance
    # optimization, we can avoid tracing this value). we trace it for
    # simplicity and uniformity of implementation.
    (typ, trace_field) = get_value_info(node)
    push!(state.stmts, quote
        $bb_new_trace.$trace_field = args_change
    end)
end

const addr_change_prefix = gensym("addrchange")

function addr_change_variable(addr::Symbol)
    Symbol("$(addr_change_prefix)_$(addr)")
end

function process!(ir::BasicBlockIR, state::Union{BBUpdateState,BBFixUpdateState,BBExtendState}, node::AddrChangeNode)
    # always mark
    push!(state.marked, node.output)

    (typ, trace_field) = get_value_info(node)
    addr = node.address
    @assert addr in state.addr_visited
    if haskey(ir.addr_dist_nodes, addr)
        dist_node = ir.addr_dist_nodes[addr]
        # TODO: this implies we cannot access @change for addresses that don't have outputs?
        constrained = dist_node.output in state.marked
        # return whether the value changed and the previous value
        push!(state.stmts, quote
            $bb_new_trace.$trace_field = $(constrained ? :(PrevChoiceDiff(trace.$addr)) : NoChoiceDiff())
        end)
    else
        if !haskey(ir.addr_gen_nodes, addr)
            # it is neither the address of a distribution or a generator
            error("Unknown address: $addr")
        end
        push!(state.stmts, quote
            $bb_new_trace.$trace_field = $(addr_change_variable(addr))
        end)
    end
end

function process_no_discard!(ir::BasicBlockIR, state::Union{BBUpdateState,BBFixUpdateState,BBExtendState}, node::AddrDistNode)
    addr = node.address
    push!(state.addr_visited, addr)
    typ = get_return_type(node.dist)
    dist = QuoteNode(node.dist)
    args = get_args(bb_new_trace, node)
    prev_args = get_args(:trace, node)
    decrement = gensym("decrement")
    increment = gensym("increment")
    input_nodes_marked = any([input in state.marked for input in node.input_nodes])
    if isa(state.schema, StaticAddressSchema) && addr in leaf_node_keys(state.schema)
        # constrained to a new value (mark the output)
        if has_output(node)
            push!(state.marked, node.output)
        end
        prev_value = gensym("prev_value")
        push!(state.stmts, quote
            $bb_new_trace.$addr = static_get_leaf_node(constraints, Val($(QuoteNode(addr)))) 
            $prev_value::$typ = trace.$addr
            $increment = logpdf($dist, $bb_new_trace.$addr, $(args...))
            $decrement = logpdf($dist, $prev_value, $(prev_args...))
            $bb_score += $increment - $decrement
            $bb_weight += $increment - $decrement
        end)
        state.discard_leaf_nodes[addr] = prev_value
        if has_output(node)
            (_, trace_field) = get_value_info(node)
            # TODO redundant with addr field, for use by other later statements:
            push!(state.stmts, quote
                $bb_new_trace.$trace_field = $bb_new_trace.$addr
            end)
        end
    elseif input_nodes_marked
         push!(state.stmts, quote
            $increment = logpdf($dist, trace.$addr, $(args...))
            $decrement = logpdf($dist, trace.$addr, $(prev_args...))
            $bb_score += $increment - $decrement
            $bb_weight += $increment - $decrement
        end)
    end
end

function process!(ir::BasicBlockIR, state::Union{BBUpdateState,BBFixUpdateState}, node::AddrDistNode)
    process_no_discard!(ir, state, node)
    if isa(state.schema, StaticAddressSchema) && addr in leaf_node_keys(state.schema)
        state.discard_leaf_nodes[addr] = prev_value
    end
end

function process!(ir::BasicBlockIR, state::BBExtendState, node::AddrDistNode)
    process_no_discard!(ir, state, node)
end

function get_constraints(schema::Union{StaticAddressSchema,EmptyAddressSchema}, addr::Symbol)
    if isa(schema, StaticAddressSchema) && addr in internal_node_keys(schema)
        constraints = :(static_get_internal_node(constraints, Val($(QuoteNode(addr)))))
        constrained = true
    else
        constrained = false
        constraints = :(EmptyAssignment())
    end
    (constrained, constraints)
end

function generate_generator_output_statement!(stmts::Vector{Expr}, node::AddrGeneratorNode, addr::Symbol)
    if has_output(node)
        (_, output_value_field) = get_value_info(node)
        push!(stmts, quote
            $bb_new_trace.$output_value_field = get_call_record($bb_new_trace.$addr).retval
        end)
    end
end

function generate_generator_call_statement!(state::BBUpdateState, addr::Symbol, node::AddrGeneratorNode, constraints)
    args = get_args(bb_new_trace, node)
    prev_args = get_args(:trace, node)
    change_value_ref = :($bb_new_trace.$(value_field(node.change_node)))
    discard = gensym("discard")
    weight = gensym("weight")
    push!(state.stmts, quote
        ($bb_new_trace.$addr, $weight, $discard, $(addr_change_variable(addr))) = update(
            $(QuoteNode(node.gen)), $(Expr(:tuple, args...)),
            $change_value_ref, trace.$addr, $constraints)
    end)
    push!(state.stmts, quote
        $bb_weight += $weight
    end)
    state.discard_internal_nodes[addr] = discard
end

function generate_generator_call_statement!(state::BBFixUpdateState, addr::Symbol, node::AddrGeneratorNode, constraints)
    args = get_args(bb_new_trace, node)
    prev_args = get_args(:trace, node)
    change_value_ref = :($bb_new_trace.$(value_field(node.change_node)))
    discard = gensym("discard")
    weight = gensym("weight")
    push!(state.stmts, quote
        ($bb_new_trace.$addr, $weight, $discard, $(addr_change_variable(addr))) = fix_update(
            $(QuoteNode(node.gen)), $(Expr(:tuple, args...)),
            $change_value_ref, trace.$addr, $constraints)
    end)
    push!(state.stmts, quote
        $bb_weight += $weight
    end)
    state.discard_internal_nodes[addr] = discard
end

function generate_generator_call_statement!(state::BBExtendState, addr::Symbol, node::AddrGeneratorNode, constraints)
    args = get_args(bb_new_trace, node)
    prev_args = get_args(:trace, node)
    change_value_ref = :($bb_new_trace.$(value_field(node.change_node)))
    weight = gensym("weight")
    push!(state.stmts, quote
        ($bb_new_trace.$addr, $weight, $(addr_change_variable(addr))) = extend(
            $(QuoteNode(node.gen)), $(Expr(:tuple, args...)),
            $change_value_ref, trace.$addr, $constraints)
    end)
    push!(state.stmts, quote
        $bb_weight += $weight
    end)
end

function generate_generator_score_statements!(stmts::Vector{Expr}, addr::Symbol)
    decrement = gensym("decrement")
    increment = gensym("increment")
    push!(stmts, quote
        $decrement = get_call_record(trace.$addr).score
        $increment = get_call_record($bb_new_trace.$addr).score
        $bb_score += $increment - $decrement
    end)
end

function process_generator_update_marked!(state::Union{BBUpdateState,BBFixUpdateState,BBExtendState}, node::AddrGeneratorNode)
    # return value could change (even if just the input nodes are marked,
    # we don't currently statically identify a generator that can absorb
    # arbitrary changes to its arguments)
    if has_output(node)
        push!(state.marked, node.output)
    end
end

function process!(ir::BasicBlockIR, state::Union{BBUpdateState,BBFixUpdateState}, node::AddrGeneratorNode)
    addr = node.address
    push!(state.addr_visited, addr)
    input_nodes_marked = any([input in state.marked for input in node.input_nodes])
    (constrained, constraints) = get_constraints(state.schema, addr)
    if constrained || input_nodes_marked
        process_generator_update_marked!(state, node)
        generate_generator_call_statement!(state, addr, node, constraints)
        generate_generator_score_statements!(state.stmts, addr)
        generate_generator_output_statement!(state.stmts, node, addr)
    else
        push!(state.stmts, quote
            $(addr_change_variable(addr)) = noargdiff
        end)
    end
end

function process!(ir::BasicBlockIR, state::BBExtendState, node::AddrGeneratorNode)
    addr = node.address
    push!(state.addr_visited, addr)
    input_nodes_marked = any([input in state.marked for input in node.input_nodes])
    (constrained, constraints) = get_constraints(state.schema, addr)
    if constrained || input_nodes_marked
        process_generator_update_marked!(state, node)
        generate_generator_call_statement!(state, addr, node, constraints)
        generate_generator_score_statements!(state.stmts, addr)
        generate_generator_output_statement!(state.stmts, node, addr)
    else
        push!(state.stmts, quote
            $(addr_change_variable(addr)) = noargdiff
        end)
    end
end

function generate_init_statements!(stmts::Vector{Expr})
    push!(stmts, quote
        $bb_new_trace = copy(trace)
        $bb_score = trace.$call_record_field.score
        $bb_weight = 0.
    end)
end

function generate_arg_statements!(stmts::Vector{Expr}, ir::BasicBlockIR)

    # unpack arguments into variables
    arg_names = Symbol[arg_node.name for arg_node in ir.arg_nodes]
    push!(stmts, Expr(:(=), Expr(:tuple, arg_names...), :new_args))

    # record arguments in trace
    for arg_node in ir.arg_nodes
        push!(stmts, quote $bb_new_trace.$(value_field(arg_node)) = $(arg_node.name) end)
    end
end

function generate_expr_node_statements!(state::Union{BBUpdateState,BBFixUpdateState,BBExtendState}, ir::BasicBlockIR)
    # visit statements in topological order, generating code for each one
    for node in ir.expr_nodes_sorted
        process!(ir, state, node)
    end
end

function generate_is_empty!(stmts::Vector{Expr}, ir::BasicBlockIR)
    # NOTE: this is still O(N) where N is the number of generator calls,
    # including non-visited calls
    if !isempty(ir.addr_dist_nodes)
        push!(stmts, quote
            $bb_new_trace.$is_empty_field = false
        end)
    else
        for (addr, node::AddrGeneratorNode) in ir.addr_gen_nodes
            push!(stmts, quote
                $bb_new_trace.$is_empty_field = $bb_new_trace.$is_empty_field && !has_choices($bb_new_trace.$addr)
            end)
        end
    end
end

function generate_discard!(stmts::Vector{Expr}, discard_leaf_nodes, discard_internal_nodes)
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
    leaf_keys = map((k) -> QuoteNode(k), leaf_keys)
    internal_keys = map((k) -> QuoteNode(k), internal_keys)
    push!(stmts, quote
        discard = StaticAssignment(
            NamedTuple{($(leaf_keys...),)}(($(leaf_nodes...),)),
            NamedTuple{($(internal_keys...),)}(($(internal_nodes...),)))
    end)
end

function check_no_extra_constraints(schema::StaticAddressSchema, ir::BasicBlockIR)
    addresses = union(keys(ir.addr_dist_nodes), keys(ir.addr_gen_nodes))
    for addr in union(leaf_node_keys(schema), internal_node_keys(schema))
        if !(addr in addresses)
            error("Update did not consume all constraints")
        end
    end
end

function check_no_extra_constraints(schema::EmptyAddressSchema, ir::BasicBlockIR)
end


function generate_call_record!(stmts::Vector{Expr}, ir::BasicBlockIR, marked::Set{ValueNode})

    # return value
    if ir.output_node === nothing
        retval = :nothing
    else
        if ir.output_node in marked
            retval = quote $bb_new_trace.$(value_field(ir.output_node)) end
        else
            retval = quote trace.$call_record_field.retval end
        end
    end

    # construct new call record
    # TODO move the change to a returnvalue of update, not part of the call record
    # if we do that, then we will need separate fields in which to store the
    # retchange values (one for each geneator)
    push!(stmts, quote
        $bb_new_trace.$call_record_field = CallRecord($bb_score, $retval, new_args)
    end)
end

function generate_update_return_statement!(stmts::Vector{Expr}, ir::BasicBlockIR)
    if ir.retchange_node === nothing
        retchange = :(GenFunctionDefaultRetDiff())
    else
        retchange = Expr(:(.), bb_new_trace, QuoteNode(value_field(ir.retchange_node)))
    end
    push!(stmts, quote return ($bb_new_trace, $bb_weight, discard, $retchange) end)
end

function generate_extend_return_statement!(stmts::Vector{Expr}, ir::BasicBlockIR)
    if ir.retchange_node === nothing
        retchange = :(GenFunctionDefaultRetDiff())
    else
        retchange = Expr(:(.), bb_new_trace, QuoteNode(value_field(ir.retchange_node)))
    end
    push!(stmts, quote return ($bb_new_trace, $bb_weight, $retchange) end)
end

function codegen_update(gen_type::Type{T}, new_args_type, args_change_type, trace_type, constraints_type) where {T <: BasicGenFunction}
    schema = get_address_schema(constraints_type)
    ir = get_ir(gen_type)
    stmts = Expr[]
    generate_init_statements!(stmts)
    generate_arg_statements!(stmts, ir)
    state = BBUpdateState(stmts, ir, schema, args_change_type)
    generate_expr_node_statements!(state, ir)
    generate_is_empty!(stmts, ir)
    generate_discard!(stmts, state.discard_leaf_nodes, state.discard_internal_nodes)
    generate_call_record!(stmts, ir, state.marked)
    generate_update_return_statement!(stmts, ir)
    return Expr(:block, stmts...)
end

function codegen_fix_update(gen_type::Type{T}, new_args_type, args_change_type, trace_type, constraints_type) where {T <: BasicGenFunction}
    schema = get_address_schema(constraints_type)
    ir = get_ir(gen_type)
    stmts = Expr[]
    generate_init_statements!(stmts)
    generate_arg_statements!(stmts, ir)
    state = BBFixUpdateState(stmts, ir, schema, args_change_type)
    generate_expr_node_statements!(state, ir)
    generate_is_empty!(stmts, ir)
    generate_discard!(stmts, state.discard_leaf_nodes, state.discard_internal_nodes)
    generate_call_record!(stmts, ir, state.marked)
    generate_update_return_statement!(stmts, ir)
    return Expr(:block, stmts...)
end

function codegen_extend(gen_type::Type{T}, new_args_type, args_change_type, trace_type, constraints_type) where {T <: BasicGenFunction}
    schema = get_address_schema(constraints_type)
    ir = get_ir(gen_type)
    stmts = Expr[]
    generate_init_statements!(stmts)
    generate_arg_statements!(stmts, ir)
    state = BBExtendState(stmts, ir, schema, args_change_type)
    generate_expr_node_statements!(state, ir)
    generate_is_empty!(stmts, ir)
    generate_call_record!(stmts, ir, state.marked)
    generate_extend_return_statement!(stmts, ir)
    return Expr(:block, stmts...)
end


push!(Gen.generated_functions, quote
@generated function Gen.update(gen::Gen.BasicGenFunction{T,U}, new_args, args_change::Union{Gen.NoArgDiff,Gen.UnknownArgDiff,Gen.MaskedArgChange}, trace::U, constraints) where {T,U}
    schema = get_address_schema(constraints)
    if !(isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema))
        # try to convert it to a static assignment
        return quote update(gen, new_args, args_change, trace, StaticAssignment(constraints)) end
    end
    Gen.codegen_update(gen, new_args, args_change, trace, constraints)
end
end)

push!(Gen.generated_functions, quote
@generated function Gen.fix_update(gen::Gen.BasicGenFunction{T,U}, new_args, args_change::Union{Gen.NoArgDiff,Gen.UnknownArgDiff,Gen.MaskedArgChange}, trace::U, constraints) where {T,U}
    schema = get_address_schema(constraints)
    if !(isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema))
        # try to convert it to a static assignment
        return quote fix_update(gen, new_args, args_change, trace, StaticAssignment(constraints)) end
    end
    Gen.codegen_fix_update(gen, new_args, args_change, trace, constraints)
end
end)

push!(Gen.generated_functions, quote
@generated function Gen.extend(gen::Gen.BasicGenFunction{T,U}, new_args, args_change::Union{Gen.NoArgDiff,Gen.UnknownArgDiff,Gen.MaskedArgChange}, trace::U, constraints) where {T,U}
    schema = get_address_schema(constraints)
    if !(isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema))
        # try to convert it to a static assignment
        return quote extend(gen, new_args, args_change, trace, StaticAssignment(constraints)) end
    end
    Gen.codegen_extend(gen, new_args, args_change, trace, constraints)
end
end)
