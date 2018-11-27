function get_value_fieldname(node::ArgumentNode)
    Symbol("arg_$(node.name)")
end

function get_value_fieldname(node::RandomChoiceNode)
    Symbol("choice_value_$(node.addr)")
end

function get_score_fieldname(node::RandomChoiceNode)
    Symbol("choice_score_$(node.addr)")
end

function get_subtrace_fieldname(node::GenerativeFunctionCallNode)
    Symbol("subtrace_$(node.addr)")
end

const num_has_choices_fieldname = Symbol("num_has_choices")

const total_score_fieldname = Symbol("score")

const return_value_fieldname = Symbol("retval")

struct TraceField
    fieldname::Symbol
    typ::Type
end

function get_trace_fields(ir::StaticIR)
    fields = TraceField[]
    for node in ir.arg_nodes
        fieldname = get_value_fieldname(node)
        push!(fields, TraceField(fieldname, node.typ))
    end
    for node in ir.choice_nodes
        value_fieldname = get_value_fieldname(node)
        push!(fields, TraceField(value_fieldname, node.typ))
        score_fieldname = get_score_fieldname(node)
        push!(fields, TraceField(score_fieldname, Float64))
    end
    for node in ir.call_nodes
        subtrace_fieldname = get_subtrace_fieldname(node)
        subtrace_type = get_trace_type(node.generative_function)
        push!(fields, TraceField(subtrace_fieldname, subtrace_type))
    end
    push!(fields, TraceField(total_score_fieldname, Float64))
    push!(fields, TraceField(num_has_choices_fieldname, Int))
    push!(fields, TraceField(return_value_fieldname, ir.return_node.typ))
    return fields
end

function generate_trace_struct(ir::StaticIR, name::Symbol)
    trace_struct_name = gensym("StaticIRTrace_$name")
    mutable = false
    fields = get_trace_fields(ir)
    field_exprs = map((f) -> Expr(:(::), f.fieldname, QuoteNode(f.typ)), fields)

    # definition of the struct type
    defn_expr = Expr(:struct, mutable, trace_struct_name,
                              Expr(:block, field_exprs...))

    # has_choices method definition
    has_choices_expr = Expr(:function,
        Expr(:call, :(Gen.has_choices), :(trace::$trace_struct_name)),
        :(trace.$num_has_choices_fieldname > 0))

    # get_call_record method definition
    args = Expr(:tuple, [:(trace.$(get_value_fieldname(node)))
                         for node in ir.arg_nodes]...)
    get_call_record_expr = Expr(:function,
        Expr(:call, :(Gen.get_call_record), :(trace::$trace_struct_name)),
        Expr(:block,
            :(score = trace.$total_score_fieldname),
            :(args = $args),
            :(retval = trace.$return_value_fieldname),
            :(CallRecord(score, retval, args)) # TODO type parameter in constructor?
        ))
    

    # TODO define the trace interface methods
    # - get_assignment()

    # - get_address_schema()

    return Expr(:block, defn_expr, has_choices_expr, get_call_record_expr)
end




