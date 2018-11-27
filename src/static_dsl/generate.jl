const trace = gensym("trace")
const weight = gensym("weight")
const subtrace = gensym("subtrace")

struct StaticIRGenerateState
    schema::Union{StaticAddressSchema, EmptyAddressSchema}
    stmts::Vector{Any}
end

function process!(::StaticIR, state::StaticIRGenerateState, node::ArgumentNode)
    push!(state.stmts, :($(get_value_fieldname(node)) = $(node.name)))
end

function process!(::StaticIR, ::StaticIRGenerateState, ::ConstantNode) end

function process!(::StaticIR, ::StaticIRGenerateState, ::DiffJuliaNode) end

function process!(::StaticIR, ::StaticIRGenerateState, ::ReceivedArgDiffNode) end

function process!(::StaticIR, ::StaticIRGenerateState, ::ChoiceDiffNode) end

function process!(::StaticIR, ::StaticIRGenerateState, ::CallDiffNode) end

function process!(::StaticIR, state::StaticIRGenerateState, node::JuliaNode)
    push!(state.stmts, :($(node.name) = $(node.expr)))
end

function process!(::StaticIR, state::StaticIRGenerateState, node::RandomChoiceNode)
    schema = state.schema
    args = map((input_node) -> input_node.name, node.inputs)
    incr = gensym("logpdf")
    addr = QuoteNode(node.addr)
    dist = QuoteNode(node.dist)
    @assert isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema)
    if isa(schema, StaticAddressSchema) && (node.addr in leaf_node_keys(schema))
        push!(state.stmts, :($(node.name) = static_get_leaf_node(constraints, Val($addr))))
        push!(state.stmts, :($incr = logpdf($dist, $(node.name), $(args...))))
        push!(state.stmts, :($weight += $incr))
    else
        push!(state.stmts, :($(node.name) = random($dist, $(args...))))
        push!(state.stmts, :($incr = logpdf($dist, $(node.name), $(args...))))
    end
    push!(state.stmts, :($(get_value_fieldname(node)) = $(node.name)))
    push!(state.stmts, :($(get_score_fieldname(node)) = $incr))
    push!(state.stmts, :($num_has_choices_fieldname += 1))
    push!(state.stmts, :($total_score_fieldname += $incr))
end

function process!(::StaticIR, state::StaticIRGenerateState, node::GenerativeFunctionCallNode)
    schema = state.schema
    args = map((input_node) -> input_node.name, node.inputs)
    args_tuple = Expr(:tuple, args...)
    addr = QuoteNode(node.addr)
    gen_fn = QuoteNode(node.generative_function)
    @assert isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema)
    subtrace = get_subtrace_fieldname(node)
    incr = gensym("weight")
    subconstraints = gensym("subconstraints")
    if isa(schema, StaticAddressSchema) && (addr in internal_node_keys(schema))
        push!(state.stmts, :($subconstraints = static_get_internal_node(constraints, Val($addr))))
        push!(state.stmts, :(($subtrace, $incr) = generate($gen_fn, $args_tuple, $subconstraints)))
        push!(state.stmts, :($weight += $incr))
    else
        push!(state.stmts, :($subtrace = simulate($gen_fn, $args_tuple)))
    end
    push!(state.stmts, :($num_has_choices_fieldname += has_choices($subtrace) ? 1 : 0))
    push!(state.stmts, :($total_score_fieldname += get_call_record($subtrace).score))
    push!(state.stmts, :($(node.name) = get_call_record($subtrace).retval))
end

function codegen_generate(gen_fn::Type{T}, args, constraints) where {T <: StaticIRGenerativeFunction}
    trace_type = get_trace_type(gen_fn)
    schema = get_address_schema(constraints)

    # convert the constraints to a static assignment if it is not already one
    if !(isa(schema, StaticAddressSchema) || isa(schema, EmptyAddressSchema))
        return quote generate(gen_fn, args, StaticAssignment(constraints)) end
    end

    ir = get_ir(gen_fn)
    stmts = []

    # initialize score, weight, and num_has_choices
    push!(stmts, :($total_score_fieldname = 0.))
    push!(stmts, :($weight = 0.))
    push!(stmts, :($num_has_choices_fieldname = 0))

    # unpack arguments
    arg_names = Symbol[arg_node.name for arg_node in ir.arg_nodes]
    push!(stmts, :($(Expr(:tuple, arg_names...)) = args))

    # process expression nodes in topological order
    state = StaticIRGenerateState(schema, stmts)
    for node in ir.nodes
        process!(ir, state, node)
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
@generated function Gen.generate(gen_fn::Gen.StaticIRGenerativeFunction, args, constraints)
    Gen.codegen_generate(gen_fn, args, constraints)
end
end)
