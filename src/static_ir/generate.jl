struct StaticIRGenerateState
    schema::Union{StaticAddressSchema, EmptyAddressSchema}
    stmts::Vector{Any}
end

function process!(::StaticIRGenerateState, node, options) end

function process!(state::StaticIRGenerateState, node::TrainableParameterNode, options)
    push!(state.stmts, :($(node.name) = $(QuoteNode(get_param))(gen_fn, $(QuoteNode(node.name)))))
end

function process!(state::StaticIRGenerateState, node::ArgumentNode, options)
    push!(state.stmts, :($(get_value_fieldname(node)) = $(node.name)))
end

function process!(state::StaticIRGenerateState, node::JuliaNode, options)
    args = map((input_node) -> input_node.name, node.inputs)
    push!(state.stmts, :($(node.name) = $(QuoteNode(node.fn))($(args...))))
    if options.cache_julia_nodes
        push!(state.stmts, :($(get_value_fieldname(node)) = $(node.name)))
    end
end

function process!(state::StaticIRGenerateState, node::GenerativeFunctionCallNode, options)
    schema = state.schema
    args = map((input_node) -> input_node.name, node.inputs)
    args_tuple = Expr(:tuple, args...)
    addr = QuoteNode(node.addr)
    gen_fn = QuoteNode(node.generative_function)
    @assert isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema)
    subtrace = get_subtrace_fieldname(node)
    incr = gensym("weight")
    subconstraints = gensym("subconstraints")
    if isa(schema, StaticAddressSchema) && (node.addr in keys(schema))
        push!(state.stmts, :($subconstraints = $(GlobalRef(Gen, :static_get_submap))(constraints, Val($addr))))
        push!(state.stmts, :(($subtrace, $incr) = $(GlobalRef(Gen, :generate))($gen_fn, $args_tuple, $subconstraints)))
    else
        push!(state.stmts, :(($subtrace, $incr) = $(GlobalRef(Gen, :generate))($gen_fn, $args_tuple, $(GlobalRef(Gen, :EmptyChoiceMap))())))
    end
    push!(state.stmts, :($weight += $incr))
    push!(state.stmts, :($num_nonempty_fieldname += !$(GlobalRef(Gen, :isempty))($(GlobalRef(Gen, :get_choices))($subtrace)) ? 1 : 0))
    push!(state.stmts, :($(node.name) = $(GlobalRef(Gen, :get_retval))($subtrace)))
    push!(state.stmts, :($total_score_fieldname += $(GlobalRef(Gen, :get_score))($subtrace)))
    push!(state.stmts, :($total_noise_fieldname += $(GlobalRef(Gen, :project))($subtrace, $(GlobalRef(Gen, :EmptySelection))())))
end

function codegen_generate(gen_fn_type::Type{T}, args,
                            constraints_type) where {T <: StaticIRGenerativeFunction}
    trace_type = get_trace_type(gen_fn_type)
    schema = get_address_schema(constraints_type)

    # convert the constraints to a static assignment if it is not already one
    if !(isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema))
        return quote $(GlobalRef(Gen, :generate))(gen_fn, args, $(QuoteNode(StaticChoiceMap))(constraints)) end
    end

    ir = get_ir(gen_fn_type)
    options = get_options(gen_fn_type)
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
    state = StaticIRGenerateState(schema, stmts)
    for node in ir.nodes
        process!(state, node, options)
    end

    # return value
    push!(stmts, :($return_value_fieldname = $(ir.return_node.name)))

    # construct trace
    push!(stmts, :($static_ir_gen_fn_ref = gen_fn))
    push!(stmts, :($trace = $(QuoteNode(trace_type))($(fieldnames(trace_type)...))))

    # return trace and weight
    push!(stmts, :(return ($trace, $weight)))

    Expr(:block, stmts...)
end
