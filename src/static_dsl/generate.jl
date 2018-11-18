struct BasicBlockGenerateState
    trace::Symbol
    score::Symbol
    weight::Symbol
    schema::Union{StaticAddressSchema, EmptyAddressSchema}
    stmts::Vector{Any}
end

function process!(ir::BasicBlockIR, state::BasicBlockGenerateState, node::JuliaNode)
    trace = state.trace
    (_, trace_field) = get_value_info(node)
    if node.line !== nothing
        push!(state.stmts, node.line)
    end
    push!(state.stmts, Expr(:(=),
        Expr(:(.), trace, QuoteNode(trace_field)),
        expr_read_from_trace(node, trace)))
end

function process!(ir::BasicBlockIR, state::BasicBlockGenerateState,
                  node::Union{ArgsChangeNode,AddrChangeNode})
    trace = state.trace
    (_, trace_field) = get_value_info(node)
    push!(state.stmts, Expr(:(=), Expr(:(.), trace, QuoteNode(trace_field)), QuoteNode(NoChoiceDiff())))
end

function process!(ir::BasicBlockIR, state::BasicBlockGenerateState, node::AddrDistNode)
    trace, score, weight, schema = (state.trace, state.score, state.weight, state.schema)
    addr = node.address
    dist = QuoteNode(node.dist)
    args = get_args(trace, node)
    if isa(schema, StaticAddressSchema) && (addr in leaf_node_keys(schema))
        increment = gensym("logpdf")
        push!(state.stmts, node.line)
        push!(state.stmts, Expr(:(=), Expr(:(.), trace, QuoteNode(addr)), Expr(:call, :static_get_leaf_node, :constraints, Expr(:call, :Val, QuoteNode(addr)))))
        push!(state.stmts, node.line)
        push!(state.stmts, Expr(:(=), increment, Expr(:call, :logpdf, dist, Expr(:(.), trace, QuoteNode(addr)), args...)))
        push!(state.stmts, node.line)
        push!(state.stmts, Expr(:(+=), score, increment))
        push!(state.stmts, node.line)
        push!(state.stmts, Expr(:(+=), weight, increment))
        push!(state.stmts, node.line)
        push!(state.stmts, Expr(:(=), Expr(:(.), trace, QuoteNode(is_empty_field)), QuoteNode(false)))
    elseif isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema)
        push!(state.stmts, node.line)
        push!(state.stmts, Expr(:(=), Expr(:(.), trace, QuoteNode(addr)), Expr(:call, :random, dist, args...)))
        push!(state.stmts, node.line)
        push!(state.stmts, Expr(:(+=), score, Expr(:call, :logpdf, dist, Expr(:(.), trace, QuoteNode(addr)), args...)))
    else
        error("Basic block does not currently support $schema constraints")
    end
    push!(state.stmts, node.line)
    push!(state.stmts, Expr(:(=), Expr(:(.), trace, QuoteNode(is_empty_field)), QuoteNode(false)))
    if has_output(node)
        (_, trace_field) = get_value_info(node)
        push!(state.stmts, node.line)
        push!(state.stmts, Expr(:(=), Expr(:(.), trace, QuoteNode(trace_field)), Expr(:(.), trace, QuoteNode(addr))))
    end
end

function process!(ir::BasicBlockIR, state::BasicBlockGenerateState, node::AddrGenerativeFunctionNode)
    trace, score, weight, schema = (state.trace, state.score, state.weight, state.schema)
    addr = node.address
    gen = QuoteNode(node.gen)
    args = get_args(trace, node)
    call_record = gensym("call_record")
    if isa(schema, StaticAddressSchema) && (addr in internal_node_keys(schema))
        weight_incr = gensym("weight")
        push!(state.stmts, node.line)
        push!(state.stmts, Expr(:(=),
            Expr(:tuple, Expr(:(.), trace, QuoteNode(addr)), weight_incr),
            Expr(:call, :generate, gen, Expr(:tuple, args...), Expr(:call, :static_get_internal_node, :constraints, Expr(:call, :Val, QuoteNode(addr))))))
        push!(state.stmts, node.line)
        push!(state.stmts, Expr(:(+=), weight, weight_incr))
    elseif isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema)
        push!(state.stmts, node.line)
        push!(state.stmts, Expr(:(=), Expr(:(.), trace, QuoteNode(addr)), Expr(:call, :simulate, gen, Expr(:tuple, args...))))
    else
        err = "Basic block does not currently support constraints: " * String(schema)
        error(err)
    end
    push!(state.stmts, node.line)
    push!(state.stmts, Expr(:(=), call_record, Expr(:call, :get_call_record, Expr(:(.), trace, QuoteNode(addr)))))
    push!(state.stmts, node.line)
    push!(state.stmts, Expr(:(+=), score, Expr(:(.), call_record, QuoteNode(:score))))
    push!(state.stmts, node.line)
    push!(state.stmts, Expr(:(=),
        Expr(:(.), trace, QuoteNode(is_empty_field)),
        Expr(:(&&), Expr(:(.), trace, QuoteNode(is_empty_field)), Expr(:call, :!, Expr(:call, :has_choices, Expr(:(.), trace, QuoteNode(addr)))))))
    if has_output(node)
        (_, trace_field) = get_value_info(node)
        push!(state.stmts, node.line)
        push!(state.stmts, Expr(:(=), Expr(:(.), trace, QuoteNode(trace_field)), Expr(:(.), call_record, QuoteNode(:retval))))
    end
end

function codegen_generate(gen::Type{T}, args, constraints) where {T <: StaticDSLFunction}
    trace_type = get_trace_type(gen)
    schema = get_address_schema(constraints)
    if !(isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema))
        # trie to convert it to a static assignment
        return quote generate(gen, args, StaticAssignment(constraints)) end
    end

    ir = get_ir(gen)
    stmts = []

    # initialize trace and score and weight
    trace = gensym("trace")
    score = gensym("score")
    weight = gensym("weight")
    push!(stmts, Expr(:(=), trace, Expr(:call, trace_type)))
    push!(stmts, Expr(:(=), score, QuoteNode(0.)))
    push!(stmts, Expr(:(=), weight, QuoteNode(0.)))
    push!(stmts, Expr(:(=), Expr(:(.), trace, QuoteNode(is_empty_field)), QuoteNode(true)))

    # unpack arguments
    arg_names = Symbol[arg_node.name for arg_node in ir.arg_nodes]
    push!(stmts, Expr(:(=), Expr(:tuple, arg_names...), :args))

    # record arguments in trace
    for arg_node in ir.arg_nodes
        push!(stmts, Expr(:(=), Expr(:(.), trace, QuoteNode(value_field(arg_node))), arg_node.name))
    end

    # process expression nodes in topological order
    state = BasicBlockGenerateState(trace, score, weight, schema, stmts)
    for node in ir.expr_nodes_sorted
        # skip incremental nodes, which are not needed (these fields will
        # remain uninitialized in the trace)
        process!(ir, state, node)
    end

    # return value
    if ir.output_node === nothing
        retval = :nothing
    else
        retval = Expr(:(.), trace, QuoteNode(value_field(something(ir.output_node))))
    end

    push!(stmts, Expr(:(=), Expr(:(.), trace, QuoteNode(call_record_field)), Expr(:call, :CallRecord, score, retval, :args)))
    push!(stmts, Expr(:return, Expr(:tuple, trace, weight)))
    Expr(:block, stmts...)
end

push!(Gen.generated_functions, quote
@generated function Gen.generate(gen::Gen.StaticDSLFunction, args, constraints)
    Gen.codegen_generate(gen, args, constraints)
end
end)
