struct StaticIRSimulateState
    stmts::Vector{Any}
end

function process!(::StaticIRSimulateState, node) end

function process!(state::StaticIRSimulateState, node::ArgumentNode)
    push!(state.stmts, :($(get_value_fieldname(node)) = $(node.name)))
end

function process!(state::StaticIRSimulateState, node::JuliaNode)
    args = map((input_node) -> input_node.name, node.inputs)
    push!(state.stmts, :($(node.name) = $(QuoteNode(node.fn))($(args...))))
end

function process!(state::StaticIRSimulateState, node::RandomChoiceNode)
    args = map((input_node) -> input_node.name, node.inputs)
    incr = gensym("logpdf")
    addr = QuoteNode(node.addr)
    dist = QuoteNode(node.dist)
    push!(state.stmts, :($(node.name) = $qn_random($dist, $(args...))))
    push!(state.stmts, :($incr = $qn_logpdf($dist, $(node.name), $(args...))))
    push!(state.stmts, :($(get_value_fieldname(node)) = $(node.name)))
    push!(state.stmts, :($(get_score_fieldname(node)) = $incr))
    push!(state.stmts, :($num_nonempty_fieldname += 1))
    push!(state.stmts, :($total_score_fieldname += $incr))
end

function process!(state::StaticIRSimulateState, node::GenerativeFunctionCallNode)
    args = map((input_node) -> input_node.name, node.inputs)
    args_tuple = Expr(:tuple, args...)
    addr = QuoteNode(node.addr)
    gen_fn = QuoteNode(node.generative_function)
    subtrace = get_subtrace_fieldname(node)
    incr = gensym("weight")
    push!(state.stmts, :($subtrace = $(QuoteNode(simulate))($gen_fn, $args_tuple)))
    push!(state.stmts, :($num_nonempty_fieldname += !$qn_isempty($qn_get_choices($subtrace)) ? 1 : 0))
    push!(state.stmts, :($(node.name) = $qn_get_retval($subtrace)))
    push!(state.stmts, :($total_score_fieldname += $qn_get_score($subtrace)))
    push!(state.stmts, :($total_noise_fieldname += $qn_project($subtrace, $qn_empty_address_set)))
end

function codegen_simulate(gen_fn_type::Type{T}, args) where {T <: StaticIRGenerativeFunction}

    ir = get_ir(gen_fn_type)
    stmts = []

    # initialize score, weight, and num_nonempty
    push!(stmts, :($total_score_fieldname = 0.))
    push!(stmts, :($total_noise_fieldname = 0.))
    push!(stmts, :($num_nonempty_fieldname = 0))

    # unpack arguments
    arg_names = Symbol[arg_node.name for arg_node in ir.arg_nodes]
    push!(stmts, :($(Expr(:tuple, arg_names...)) = args))

    # process expression nodes in topological order
    state = StaticIRSimulateState(stmts)
    for node in ir.nodes
        process!(state, node)
    end

    # return value
    push!(stmts, :($return_value_fieldname = $(ir.return_node.name)))

    # construct trace
    trace_type = get_trace_type(gen_fn_type)
    push!(stmts, :($trace = $(QuoteNode(trace_type))($(fieldnames(trace_type)...))))

    # return trace
    push!(stmts, :(return $trace))

    Expr(:block, stmts...)
end

push!(generated_functions, quote
@generated function $(Expr(:(.), Gen, QuoteNode(:simulate)))(gen_fn::$(QuoteNode(StaticIRGenerativeFunction)), args::Tuple)
    $(QuoteNode(codegen_simulate))(gen_fn, args)
end
end)
