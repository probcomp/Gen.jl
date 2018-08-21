struct BBUpdateState
    score::Symbol
    weight::Symbol
    new_trace::Symbol
    prev_trace::Symbol
    marked::Set{ValueNode}
    stmts::Vector{Expr}
    schema::Union{StaticAddressSchema,EmptyAddressSchema}
    addr_visited::Set{Symbol}
    discard_leaf_nodes::Dict{Symbol, Symbol} # map from address to value
    discard_internal_nodes::Dict{Symbol, Symbol} # map from address to value
end

function process!(ir::BasicBlockIR, state::BBUpdateState, node::JuliaNode)
    
    # if any input nodes are marked, mark the output node
    if any([input in state.marked for input in node.input_nodes])
        push!(state.marked, node.output)
    end

    # set the value in the new trace based on other values in the new trace
    (typ, trace_field) = get_value_info(node)
    if node.output in state.marked
        push!(state.stmts, quote
            $(state.new_trace).$trace_field = $(expr_read_from_trace(node, state.new_trace))
        end)
    end
end

function process!(ir::BasicBlockIR, state::BBUpdateState, node::ArgsChangeNode)
    # always mark
    push!(state.marked, node.output)

    # set the value in the new trace (in the future, for performance
    # optimization, we can avoid tracing this value). we trace it for
    # simplicity and uniformity of implementation.
    (typ, trace_field) = get_value_info(node)
    push!(state.stmts, quote
        $(state.new_trace).$trace_field = args_change
    end)
end

const addr_change_prefix = gensym("addrchange")

function addr_change_variable(addr::Symbol)
    Symbol("$(addr_change_prefix)_$(addr)")
end

function process!(ir::BasicBlockIR, state::BBUpdateState, node::AddrChangeNode)
    # always mark
    push!(state.marked, node.output)

    (typ, trace_field) = get_value_info(node)
    addr = node.address
    new_trace, prev_trace = state.new_trace, state.prev_trace
    @assert addr in state.addr_visited
    if haskey(ir.addr_dist_nodes, addr)
        dist_node = ir.addr_dist_nodes[addr]
        # TODO: this implies we cannot access @change for addresses that don't have outputs?
        constrained = dist_node.output in state.marked
        # return whether the value changed and the previous value
        push!(state.stmts, quote
            $new_trace.$trace_field = ($(QuoteNode(constrained)), $prev_trace.$addr)
        end)
    else
        if !haskey(ir.addr_gen_nodes, addr)
            # it is neither the address of a distribution or a generator
            error("Unknown address: $addr")
        end
        push!(state.stmts, quote
            $new_trace.$trace_field = $(addr_change_variable(addr))
        end)
    end
end

function process!(ir::BasicBlockIR, state::BBUpdateState, node::AddrDistNode)
    new_trace, prev_trace = state.new_trace, state.prev_trace
    score, weight = state.score, state.weight
    schema = state.schema
    addr = node.address
    push!(state.addr_visited, addr)
    typ = get_return_type(node.dist)
    dist = QuoteNode(node.dist)
    args = get_args(new_trace, node)
    prev_args = get_args(prev_trace, node)
    decrement = gensym("decrement")
    increment = gensym("increment")
    input_nodes_marked = any([input in state.marked for input in node.input_nodes])
    if isa(schema, StaticAddressSchema) && addr in keys(schema)
        # constrained to a new value (mark the output)
        if has_output(node)
            push!(state.marked, node.output)
        end
        if !schema[addr].is_primitive
            error("Expected primitive address but got namespace at $addr")
        end
        prev_value = gensym("prev_value")
        push!(state.stmts, quote
            $new_trace.$addr = get_leaf_node(constraints, Val($(QuoteNode(addr)))) 
            $prev_value::$typ = $prev_trace.$addr
            $increment = logpdf($dist, $new_trace.$addr, $(args...))
            $decrement = logpdf($dist, $prev_value, $(prev_args...))
            $score += $increment - $decrement
            $weight += $increment - $decrement
            #GenLite.set_leaf_node!(discard, Val($(QuoteNode(addr))), $prev_value) # TODO populate it later
        end)
        state.discard_leaf_nodes[addr] = prev_value
        if has_output(node)
            (_, trace_field) = get_value_info(node)
            # TODO redundant with addr field, for use by other later statements:
            push!(state.stmts, quote
                $new_trace.$trace_field = $new_trace.$addr
            end)
        end
    elseif input_nodes_marked
         push!(state.stmts, quote
            $increment = logpdf($dist, $prev_trace.$addr, $(args...))
            $decrement = logpdf($dist, $prev_trace.$addr, $(prev_args...))
            $score += $increment - $decrement
            $weight += $increment - $decrement
        end)
    end
end

function process!(ir::BasicBlockIR, state::BBUpdateState, node::AddrGeneratorNode)
    new_trace, prev_trace = state.new_trace, state.prev_trace
    score, weight = state.score, state.weight
    schema = state.schema
    addr = node.address
    push!(state.addr_visited, addr)
    args = get_args(new_trace, node)
    prev_args = get_args(prev_trace, node)
    decrement = gensym("decrement")
    increment = gensym("increment")
    call_record = gensym("call_record")
    discard = gensym("discard")
    input_nodes_marked = any([input in state.marked for input in node.input_nodes])
    if isa(schema, StaticAddressSchema) && addr in keys(schema)
        if schema[addr].is_primitive
            error("Expected namespace but got primitive address at $addr")
        end
        constraints = :(get_internal_node(constraints, Val($(QuoteNode(addr)))))
        constrained = true
    else
        constrained = false
        constraints = :(EmptyChoiceTrie())
    end
    if constrained || input_nodes_marked
        # return value could change (even if just the input nodes are marked,
        # we don't currently statically identify a generator that can absorb
        # arbitrary changes to its arguments)
        if has_output(node)
            push!(state.marked, node.output)
        end
        change_value_ref = :($new_trace.$(value_field(node.change_node)))
        #change = :(isnull($change_value_ref) ? nothing : get($change_value_ref))
        push!(state.stmts, quote
            ($new_trace.$addr, _, $discard, $(addr_change_variable(addr))) = update(
                $(QuoteNode(node.gen)), $(Expr(:tuple, args...)),
                $change_value_ref, $prev_trace.$addr, $constraints, read_trace)
                #GenLite.get_internal_node_proto(discard, Val($(QuoteNode(addr))))) # TODO deleted
            $call_record = get_call_record($new_trace.$addr)
            $decrement = get_call_record($prev_trace.$addr).score
            $increment = $call_record.score
            $score += $increment - $decrement
            $weight += $increment - $decrement
            #GenLite.set_internal_node!(discard, Val($(QuoteNode(addr))), $discard) # TODO popuilate it later
        end)
        state.discard_internal_nodes[addr] = discard # TODO
        if has_output(node)
            (_, trace_field) = get_value_info(node)
            push!(state.stmts, quote
                $new_trace.$trace_field = $call_record.retval
            end)
        end
    else
        push!(state.stmts, quote
            $(addr_change_variable(addr)) = NoChange()
        end)
    end
end

####
function mark_arguments!(marked, arg_nodes, args_change::Type{Nothing})
    for arg_node in arg_nodes
        push!(marked, arg_node)
    end
end
function mark_arguments!(marked, arg_nodes, args_change::Type{NoChange}) end
function mark_arguments!(marked, arg_nodes, args_change::Type{T}) where {T <: MaskedArgChange}
    mask = args_change.parameters[1].parameters
    for (arg_node, maybe_changed_val) in zip(arg_nodes, mask)
        if maybe_changed_val.parameters[1]
            push!(marked, arg_node)
        end
    end
end


function codegen_update(gen::Type{T}, new_args, args_change, trace, constraints, read_trace) where {T <: BasicGenFunction}
    Core.println("generating update($gen, args_change:$args_change, constraints: $constraints...)")

    trace_type = get_trace_type(gen)
    schema = get_address_schema(constraints)
    if !(isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema))
        return update(gen, new_args, args_change, trace, StaticChoiceTrie(constraints), read_trace)
        #error("Address schema of type $schema not supported for constraints of basic block")
    end
    ir = get_ir(gen)
    stmts = Expr[]
    
    # unpack arguments
    arg_names = Symbol[arg_node.name for arg_node in ir.arg_nodes]
    push!(stmts, Expr(:(=), Expr(:tuple, arg_names...), :new_args))

    score = gensym("score")
    weight = gensym("weight")
    new_trace = gensym("trace")
    prev_trace = :trace

    push!(stmts, quote
        $new_trace = copy($prev_trace)
        $score = $prev_trace.$call_record_field.score
        $weight = 0.
        discard = $discard_proto()
    end)

    # record arguments in trace
    for arg_node in ir.arg_nodes
        push!(stmts, quote $new_trace.$(value_field(arg_node)) = $(arg_node.name) end)
    end

    addr_visited = Set{Symbol}()
    marked = Set{ValueNode}()
    mark_arguments!(marked, ir.arg_nodes, args_change)

    for node in ir.generator_input_change_nodes
        # for now, mark every input change node to a generator
        # TODO we should only mark the node if the corresponding generator is
        # either constrained or marked, however, this requires two passes.
        # postponed for simplicity.
        push!(marked, node)
    end

    # visit statements in topological order, generating code
    state = BBUpdateState(score, weight, new_trace, prev_trace, marked,
                          stmts, schema, discard_proto, addr_visited)
    for node in ir.expr_nodes_sorted
        process!(ir, state, node)
    end

    # compute is_empty (NOTE: this is still O(N) where N is the number of
    # generator calls, including non-visited calls)
    if !isempty(ir.addr_dist_nodes)
        push!(stmts, quote
            $new_trace.$is_empty_field = false
        end)
    else
        for (addr, node::AddrGeneratorNode) in ir.addr_gen_nodes
            push!(stmts, quote
                $new_trace.$is_empty_field = $new_trace.$is_empty_field && !has_choices($new_trace.$addr)
            end)
        end
    end

    # constructed discard
    # TODO
    discard_leaf_node_keys = collect(keys(state.discard_leaf_nodes))
    discard_leaf_node_values = Symbol[state.discard_leaf_nodes[key] for key in discard_leaf_node_keys]
    discard_internal_node_keys = collect(keys(state.discard_internal_nodes))
    #discard_internal_node_keys = Symbol[state.discard_leaf_nodes[key] for key in discard_leaf_node_keys]
    push!(stmts, quote
        leaf_nodes = NamedTuple{($(discard_leaf_node_keys),)}()
        StaticChoiceTrie{()}()
    end)

    # check that there are no extra constraints
    if isa(schema, StaticAddressSchema)
        addresses = union(keys(ir.addr_dist_nodes), keys(ir.addr_gen_nodes))
        for addr in keys(schema)
            if !(addr in addresses)
                error("Update did not consume all constraints")
            end
        end
    end

    # return value
    if ir.output_node === nothing
        retval = :nothing
    else
        if ir.output_node in marked
            retval = quote $new_trace.$(value_field(ir.output_node)) end
        else
            retval = quote $prev_trace.$call_record_field.retval end
        end
    end

    # retchange
    if ir.retchange_node === nothing
        retchange = :(nothing)
    else
        retchange = Expr(:(.), new_trace, QuoteNode(value_field(ir.retchange_node)))
    end

    # construct new call record and return
    # TODO move the change to a returnvalue of update, not part of the call record
    # if we do that, then we will need separate fields in which to store the
    # retchange values (one for each geneator)
    push!(stmts, quote
        $new_trace.$call_record_field = CallRecord($score, $retval, new_args)
        return ($new_trace, $weight, discard, $retchange)
    end)
    Expr(:block, stmts...)
end
