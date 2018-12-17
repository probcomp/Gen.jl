struct StaticIRProjectState
    schema::Union{StaticAddressSchema, EmptyAddressSchema}
    stmts::Vector{Any}
end

function process!(state::StaticIRProjectState, node) end

function process!(state::StaticIRProjectState, node::RandomChoiceNode)
    schema = state.schema
    @assert isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema)
    if isa(schema, StaticAddressSchema) && (node.addr in leaf_node_keys(schema))
        push!(state.stmts, :($weight += trace.$(get_score_fieldname(node))))
    end
end

function process!(state::StaticIRProjectState, node::GenerativeFunctionCallNode)
    schema = state.schema
    addr = QuoteNode(node.addr)
    @assert isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema)
    subtrace = get_subtrace_fieldname(node)
    subselection = gensym("subselection")
    if isa(schema, StaticAddressSchema) && (node.addr in internal_node_keys(schema))
        push!(state.stmts, :($subselection = Gen.static_get_internal_node(selection, Val($addr))))
        push!(state.stmts, :($weight += project(trace.$subtrace, $subselection)))
    else
        push!(state.stmts, :($weight += project(trace.$subtrace, EmptyAddressSet())))
    end
end

function codegen_project(trace_type::Type, selection_type::Type)
    gen_fn_type = get_gen_fn_type(trace_type)
    schema = get_address_schema(selection_type)

    # convert the selection to a static selection if it is not already one
    if !(isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema))
        return quote project(trace, StaticAddressSet(selection)) end
    end

    ir = get_ir(gen_fn_type)
    stmts = []

    # initialize weight
    push!(stmts, :($weight = 0.))

    # process expression nodes in topological order
    state = StaticIRProjectState(schema, stmts)
    for node in ir.nodes
        process!(state, node)
    end

    # return trace and weight
    push!(stmts, :(return $weight))

    Expr(:block, stmts...)
end

push!(Gen.generated_functions, quote
@generated function Gen.project(trace::StaticIRTrace, selection::AddressSet)
    Gen.codegen_project(trace, selection)
end
end)
