struct StaticIRInitializeState
    schema::Union{StaticAddressSchema, EmptyAddressSchema}
    stmts::Vector{Any}
end

function process!(::StaticIRInitializeState, node) end

function process!(state::StaticIRInitializeState, node::ArgumentNode)
    push!(state.stmts, :($(get_value_fieldname(node)) = $(node.name)))
end

function process!(state::StaticIRInitializeState, node::JuliaNode)
    args = map((input_node) -> input_node.name, node.inputs)
    push!(state.stmts, :($(node.name) = $(QuoteNode(node.fn))($(args...))))
end

function process!(state::StaticIRInitializeState, node::RandomChoiceNode)
    schema = state.schema
    args = map((input_node) -> input_node.name, node.inputs)
    incr = gensym("logpdf")
    addr = QuoteNode(node.addr)
    dist = QuoteNode(node.dist)
    @assert isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema)
    if isa(schema, StaticAddressSchema) && (node.addr in leaf_node_keys(schema))
        push!(state.stmts, :($(node.name) = static_get_value(constraints, Val($addr))))
        push!(state.stmts, :($incr = logpdf($dist, $(node.name), $(args...))))
        push!(state.stmts, :($weight += $incr))
    else
        push!(state.stmts, :($(node.name) = random($dist, $(args...))))
        push!(state.stmts, :($incr = logpdf($dist, $(node.name), $(args...))))
    end
    push!(state.stmts, :($(get_value_fieldname(node)) = $(node.name)))
    push!(state.stmts, :($(get_score_fieldname(node)) = $incr))
    push!(state.stmts, :($num_nonempty_fieldname += 1))
    push!(state.stmts, :($total_score_fieldname += $incr))
end

function process!(state::StaticIRInitializeState, node::GenerativeFunctionCallNode)
    schema = state.schema
    args = map((input_node) -> input_node.name, node.inputs)
    args_tuple = Expr(:tuple, args...)
    addr = QuoteNode(node.addr)
    gen_fn = QuoteNode(node.generative_function)
    @assert isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema)
    subtrace = get_subtrace_fieldname(node)
    incr = gensym("weight")
    subconstraints = gensym("subconstraints")
    if isa(schema, StaticAddressSchema) && (node.addr in internal_node_keys(schema))
        push!(state.stmts, :($subconstraints = static_get_subassmt(constraints, Val($addr))))
        push!(state.stmts, :(($subtrace, $incr) = generate($gen_fn, $args_tuple, $subconstraints)))
    else
        push!(state.stmts, :(($subtrace, $incr) = generate($gen_fn, $args_tuple, EmptyAssignment())))
    end
    push!(state.stmts, :($weight += $incr))
    push!(state.stmts, :($num_nonempty_fieldname += !isempty(get_assmt($subtrace)) ? 1 : 0))
    push!(state.stmts, :($(node.name) = get_retval($subtrace)))
    push!(state.stmts, :($total_score_fieldname += get_score($subtrace)))
    push!(state.stmts, :($total_noise_fieldname += project($subtrace, EmptyAddressSet())))
end

function codegen_generate(gen_fn_type::Type{T}, args,
                            constraints_type) where {T <: StaticIRGenerativeFunction}
    trace_type = get_trace_type(gen_fn_type)
    schema = get_address_schema(constraints_type)

    # convert the constraints to a static assignment if it is not already one
    if !(isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema))
        return quote generate(gen_fn, args, StaticAssignment(constraints)) end
    end

    ir = get_ir(gen_fn_type)
    stmts = []

    # initialize score, weight, and num_nonempty
    push!(stmts, :($total_score_fieldname = 0.))
    push!(stmts, :($total_noise_fieldname = 0.))
    push!(stmts, :($weight = 0.))
    push!(stmts, :($num_nonempty_fieldname = 0))

    # unpack arguments
    arg_names = Symbol[arg_node.name for arg_node in ir.arg_nodes]
    push!(stmts, :($(Expr(:tuple, arg_names...)) = args))

    # process expression nodes in topological order
    state = StaticIRInitializeState(schema, stmts)
    for node in ir.nodes
        process!(state, node)
    end

    # return value
    push!(stmts, :($return_value_fieldname = $(ir.return_node.name)))

    # construct trace
    push!(stmts, :($trace = $(QuoteNode(trace_type))($(fieldnames(trace_type)...))))

    # return trace and weight
    push!(stmts, :(return ($trace, $weight)))

    Expr(:block, stmts...)
end

push!(Gen.generated_functions, quote
@generated function Gen.generate(gen_fn::Gen.StaticIRGenerativeFunction,
                                   args::Tuple, constraints::Assignment)
    Gen.codegen_generate(gen_fn, args, constraints)
end
end)

function propose(gen_fn::StaticIRGenerativeFunction, args::Tuple)
    # TODO implement the actual propose
    (trace, weight) = generate(gen_fn, args, EmptyAssignment())
    (get_assmt(trace), weight, get_retval(trace))
end

function assess(gen_fn::StaticIRGenerativeFunction, args::Tuple, constraints::Assignment)
    # TODO implement the actual assess
    (trace, weight) = generate(gen_fn, args, constraints)
    (weight, get_retval(trace))
end
