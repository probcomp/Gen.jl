struct StaticDataFlowAssessState
    trace::Symbol
    score::Symbol
    stmts::Vector{Expr}
end

function process!(ir::DataFlowIR, state::StaticDataFlowAssessState, node::JuliaNode)
    trace = state.trace
    (typ, trace_field) = get_value_info(node)
    push!(state.stmts, quote
        $trace.$trace_field = $(expr_read_from_trace(node, trace))
    end)
end

function process!(ir::DataFlowIR, state::StaticDataFlowAssessState,
                  node::Union{ArgsChangeNode,AddrChangeNode})
    trace = state.trace
    (typ, trace_field) = get_value_info(node)
    push!(state.stmts, quote
        $trace.$trace_field = nothing
    end)
end


function process!(ir::DataFlowIR, state::StaticDataFlowAssessState, node::AddrDistNode)
    trace, score = (state.trace, state.score)
    addr = node.address
    dist = QuoteNode(node.dist)
    args = get_args(trace, node)
    value = value_trace_ref(trace, node.output)
    push!(state.stmts, quote
        $value = static_get_leaf_node(constraints, Val($(QuoteNode(addr))))
        $score += logpdf($dist, $value, $(args...))
        $trace.$is_empty_field = false
    end)
end

function process!(ir::DataFlowIR, state::StaticDataFlowAssessState, node::AddrGeneratorNode)
    trace, score = (state.trace, state.score)
    addr = node.address
    gen = QuoteNode(node.gen)
    args = get_args(trace, node)
    call_record = gensym("call_record")
    push!(state.stmts, quote
        $trace.$addr = assess($gen, $(Expr(:tuple, args...)),
            static_get_internal_node(constraints, Val($(QuoteNode(addr)))))
        $call_record = get_call_record($trace.$addr)
        $score += $call_record.score
        $trace.$is_empty_field = $trace.$is_empty_field && !has_choices($trace.$addr)
    end)
    (_, trace_field) = get_value_info(node)
    push!(state.stmts, quote
        $trace.$trace_field = $call_record.retval
    end)
end

function codegen_assess(gen::Type{T}, args, constraints) where {T <: StaticDataFlowGenerator}
    trace_type = get_trace_type(gen)
    schema = get_address_schema(constraints) # TODO use schema to check there are no extra addrs
    ir = get_ir(gen)
    stmts = Expr[]

    # initialize trace and score
    trace = gensym("trace")
    score = gensym("score")
    push!(stmts, quote
        $trace = $trace_type()
        $score = 0.
        $trace.$is_empty_field = true
    end)

    # unpack arguments
    arg_names = Symbol[arg_node.name for arg_node in ir.arg_nodes]
    push!(stmts, Expr(:(=), Expr(:tuple, arg_names...), :args))

    # record arguments in trace
    for arg_node in ir.arg_nodes
        push!(stmts, quote $trace.$(value_field(arg_node)) = $(arg_node.name) end)
    end

    # record parameters in trace
    for param in ir.params
        value_node = ir.value_nodes[param.name]
        push!(stmts, quote $trace.$(value_field(value_node)) = gen.params[$(QuoteNode(param.name))] end)
    end

    # process expression nodes in topological order
    state = StaticDataFlowAssessState(trace, score, stmts)
    for node in ir.expr_nodes_sorted
        process!(ir, state, node)
    end

    if ir.output_node === nothing
        retval = :nothing
    else
        retval = quote $trace.$(value_field(something(ir.output_node))) end
    end

    push!(stmts, quote
        $trace.$call_record_field = CallRecord($score, $retval, args)
        return $trace
    end)
    Expr(:block, stmts...)
end


push!(Gen.generated_functions, quote
@generated function Gen.assess(gen::Gen.StaticDataFlowGenerator, args, constraints)
    schema = get_address_schema(constraints)
    if !(isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema))
        # try to convert it to a static choice trie
        return quote assess(gen, args, StaticAssignment(constraints)) end
    end
    Gen.codegen_assess(gen, args, constraints)
end
end)
