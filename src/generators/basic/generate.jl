struct BasicBlockGenerateState
    trace::Symbol
    score::Symbol
    weight::Symbol
    schema::Union{StaticAddressSchema,EmptyAddressSchema}
    stmts::Vector{Expr}
end

function process!(ir::BasicBlockIR, state::BasicBlockGenerateState, node::JuliaNode)
    trace = state.trace
    (typ, trace_field) = get_value_info(node)
    push!(state.stmts, quote
        $trace.$trace_field = ($(expr_read_from_trace(node, trace)))
    end)
end

function process!(ir::BasicBlockIR, state::BasicBlockGenerateState,
                  node::Union{ArgsChangeNode,AddrChangeNode})
    trace = state.trace
    (typ, trace_field) = get_value_info(node)
    push!(state.stmts, quote
        $trace.$trace_field = nothing
    end)
end

function process!(ir::BasicBlockIR, state::BasicBlockGenerateState, node::AddrDistNode)
    trace, score, weight, schema = (state.trace, state.score, state.weight, state.schema)
    addr = node.address
    dist = QuoteNode(node.dist)
    args = get_args(trace, node)
    if isa(schema, StaticAddressSchema) && (addr in leaf_node_keys(schema))
        increment = gensym("logpdf")
        push!(state.stmts, quote
            $trace.$addr = static_get_leaf_node(constraints, Val($(QuoteNode(addr))))
            $increment = logpdf($dist, $trace.$addr, $(args...))
            $score += $increment
            $weight += $increment
            $trace.$is_empty_field = false
        end)

    elseif isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema)
        push!(state.stmts, quote
            $trace.$addr = random($dist, $(args...))
            $score += logpdf($dist, $trace.$addr, $(args...))
            $trace.$is_empty_field = false
        end)
    else
        error("Basic block does not currently support $schema constraints")
    end
    if has_output(node)
        (_, trace_field) = get_value_info(node)
        push!(state.stmts, quote
            $trace.$trace_field = $trace.$addr
        end)
    end
end

function process!(ir::BasicBlockIR, state::BasicBlockGenerateState, node::AddrGeneratorNode)
    trace, score, weight, schema = (state.trace, state.score, state.weight, state.schema)
    addr = node.address
    gen = QuoteNode(node.gen)
    args = get_args(trace, node)
    call_record = gensym("call_record")
    if isa(schema, StaticAddressSchema) && (addr in internal_node_keys(schema))
        weight_incr = gensym("weight")
        push!(state.stmts, quote
            ($trace.$addr, $weight_incr) = generate(
                $gen, $(Expr(:tuple, args...)),
                static_get_internal_node(constraints, Val($(QuoteNode(addr)))))
            $weight += $weight_incr
            $call_record = get_call_record($trace.$addr)
            $score += $call_record.score
            $trace.$is_empty_field = $trace.$is_empty_field && !has_choices($trace.$addr)
        end)
    elseif isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema)
        push!(state.stmts, quote
            $trace.$addr = simulate($gen, $(Expr(:tuple, args...)))
            $call_record = get_call_record($trace.$addr)
            $score += $call_record.score
            $trace.$is_empty_field = $trace.$is_empty_field && !has_choices($trace.$addr)
        end)
    else
        error("Basic block does not currently support $schema constraints")
    end
    if has_output(node)
        (_, trace_field) = get_value_info(node)
        push!(state.stmts, quote
            $trace.$trace_field = $call_record.retval
        end)
    end
end

function codegen_generate(gen::Type{T}, args, constraints) where {T <: BasicGenFunction}
    Core.println("generating generate($gen, constraints: $constraints...)")
    trace_type = get_trace_type(gen)
    schema = get_address_schema(constraints)
    if !(isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema))
        # trie to convert it to a static choice trie
        return quote generate(gen, args, StaticChoiceTrie(constraints)) end
    end

    ir = get_ir(gen)
    stmts = Expr[]

    # initialize trace and score and weight
    trace = gensym("trace")
    score = gensym("score")
    weight = gensym("weight")
    push!(stmts, quote
        $trace = $trace_type()
        $score = 0.
        $weight = 0.
        $trace.$is_empty_field = true
    end)

    # unpack arguments
    arg_names = Symbol[arg_node.name for arg_node in ir.arg_nodes]
    push!(stmts, Expr(:(=), Expr(:tuple, arg_names...), :args))

    # record arguments in trace
    for arg_node in ir.arg_nodes
        push!(stmts, quote $trace.$(value_field(arg_node)) = $(arg_node.name) end)
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
        retval = quote $trace.$(value_field(something(ir.output_node))) end
    end

    push!(stmts, quote
        $trace.$call_record_field = CallRecord($score, $retval, args)
        return ($trace, $weight)
    end)
    Expr(:block, stmts...)
end

push!(Gen.generated_functions, quote
@generated function Gen.generate(gen::Gen.BasicGenFunction, args, constraints)
    Gen.codegen_generate(gen, args, constraints)
end
end)
