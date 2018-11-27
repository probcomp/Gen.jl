######################
# assignment wrapper #
######################

struct StaticIRTraceAssignment{T} <: Assignment
    trace::T
end

function get_address_schema(::Type{StaticIRTraceAssignment{T}}) where {T}
    get_schema(T)
end

Base.isempty(assmt::StaticIRTraceAssignment) = !has_choices(assmt.trace)

static_has_leaf_node(assmt::StaticIRTraceAssignment, key) = false

static_has_internal_node(assmt::StaticIRTraceAssignment, key) = false

function get_leaf_node(assmt::StaticIRTraceAssignment, key::Symbol)
    static_get_leaf_node(assmt, Val(key))
end

function get_internal_node(assmt::StaticIRTraceAssignment, key::Symbol)
    static_get_internal_node(assmt, Val(key))
end

function has_leaf_node(assmt::StaticIRTraceAssignment, key::Symbol)
    static_has_leaf_node(assmt, Val(key))
end

function has_internal_node(assmt::StaticIRTraceAssignment, key::Symbol)
    static_has_internal_node(assmt, Val(key))
end

function has_leaf_node(assmt::StaticIRTraceAssignment, addr::Pair)
    _has_leaf_node(assmt, addr)
end

function get_leaf_node(assmt::StaticIRTraceAssignment, addr::Pair)
    _get_leaf_node(assmt, addr)
end

function has_internal_node(assmt::StaticIRTraceAssignment, addr::Pair)
    _has_internal_node(assmt, addr)
end

function get_internal_node(assmt::StaticIRTraceAssignment, addr::Pair)
    _get_internal_node(assmt, addr)
end


#########################
# trace type generation #
#########################

const arg_prefix = gensym("arg")
const choice_value_prefix = gensym("choice_value")
const choice_score_prefix = gensym("choice_score")
const subtrace_prefix = gensym("subtrace")

function get_value_fieldname(node::ArgumentNode)
    Symbol("$(arg_prefix)_$(node.name)")
end

function get_value_fieldname(node::RandomChoiceNode)
    Symbol("$(choice_value_prefix)_$(node.addr)")
end

function get_score_fieldname(node::RandomChoiceNode)
    Symbol("$(choice_score_prefix)_$(node.addr)")
end

function get_subtrace_fieldname(node::GenerativeFunctionCallNode)
    Symbol("$(subtrace_prefix)_$(node.addr)")
end

const num_has_choices_fieldname = gensym("num_has_choices")
const total_score_fieldname = gensym("score")
const return_value_fieldname = gensym("retval")

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

function generate_trace_struct(ir::StaticIR, trace_struct_name::Symbol)
    mutable = false
    fields = get_trace_fields(ir)
    field_exprs = map((f) -> Expr(:(::), f.fieldname, QuoteNode(f.typ)), fields)
    Expr(:struct, mutable, trace_struct_name,
         Expr(:block, field_exprs...))
end

function generate_has_choices(trace_struct_name::Symbol)
    Expr(:function,
        Expr(:call, :(Gen.has_choices), :(trace::$trace_struct_name)),
        :(trace.$num_has_choices_fieldname > 0))
end

function generate_get_call_record(ir::StaticIR, trace_struct_name::Symbol)
    args = Expr(:tuple, [:(trace.$(get_value_fieldname(node)))
                         for node in ir.arg_nodes]...)
    Expr(:function,
        Expr(:call, :(Gen.get_call_record), :(trace::$trace_struct_name)),
        Expr(:block,
            :(score = trace.$total_score_fieldname),
            :(args = $args),
            :(retval = trace.$return_value_fieldname),
            :(CallRecord(score, retval, args)) # TODO type parameter in constructor?
        ))
end

function generate_get_assignment(trace_struct_name::Symbol)
    Expr(:function,
        Expr(:call, :(Gen.get_assignment), :(trace::$trace_struct_name)),
        Expr(:block, 
            :(StaticIRTraceAssignment(trace))))
end

function generate_get_leaf_nodes(ir::StaticIR, trace_struct_name::Symbol)
    elements = []
    for node in ir.choice_nodes
        addr = node.addr
        value = :(assmt.trace.$(get_value_fieldname(node)))
        push!(elements, :(($(QuoteNode(addr)), $value)))
    end
    Expr(:function, 
        Expr(:call, :(Gen.get_leaf_nodes),
                    :(assmt::Gen.StaticIRTraceAssignment{$trace_struct_name})),
        Expr(:block, Expr(:tuple, elements...)))
end

function generate_get_internal_nodes(ir::StaticIR, trace_struct_name::Symbol)
    elements = []
    for node in ir.call_nodes
        addr = node.addr
        subtrace = :(assmt.trace.$(get_subtrace_fieldname(node)))
        push!(elements, :(($(QuoteNode(addr)), get_assignment($subtrace))))
    end
    Expr(:function, 
        Expr(:call, :(Gen.get_internal_nodes),
                    :(assmt::Gen.StaticIRTraceAssignment{$trace_struct_name})),
        Expr(:block, Expr(:tuple, elements...)))
end

function generate_static_get_leaf_node(ir::StaticIR, trace_struct_name::Symbol)
    methods = Expr[]
    for node in ir.choice_nodes
        push!(methods, Expr(:function,
            Expr(:call, :(Gen.static_get_leaf_node),
                        :(assmt::Gen.StaticIRTraceAssignment{$trace_struct_name}),
                        :(::Val{$(QuoteNode(node.addr))})),
            Expr(:block, :(assmt.trace.$(get_value_fieldname(node))))))
    end
    methods
end

function generate_static_has_leaf_node(ir::StaticIR, trace_struct_name::Symbol)
    methods = Expr[]
    for node in ir.choice_nodes
        push!(methods, Expr(:function,
            Expr(:call, :(Gen.static_has_leaf_node),
                        :(assmt::Gen.StaticIRTraceAssignment{$trace_struct_name}),
                        :(::Val{$(QuoteNode(node.addr))})),
            Expr(:block, :(true))))
    end
    methods
end

function generate_static_get_internal_node(ir::StaticIR, trace_struct_name::Symbol)
    methods = Expr[]
    for node in ir.call_nodes
        push!(methods, Expr(:function,
            Expr(:call, :(Gen.static_get_internal_node),
                        :(assmt::Gen.StaticIRTraceAssignment{$trace_struct_name}),
                        :(::Val{$(QuoteNode(node.addr))})),
            Expr(:block,
                :(get_assignment(assmt.trace.$(get_subtrace_fieldname(node)))))))
    end
    methods
end

function generate_static_has_internal_node(ir::StaticIR, trace_struct_name::Symbol)
    methods = Expr[]
    for node in ir.call_nodes
        push!(methods, Expr(:function,
            Expr(:call, :(Gen.static_has_internal_node),
                        :(assmt::Gen.StaticIRTraceAssignment{$trace_struct_name}),
                        :(::Val{$(QuoteNode(node.addr))})),
            Expr(:block, :(true))))
    end
    methods
end


function generate_get_schema(ir::StaticIR, trace_struct_name::Symbol)
    choice_addrs = [QuoteNode(node.addr) for node in ir.choice_nodes]
    call_addrs = [QuoteNode(node.addr) for node in ir.call_nodes]
    Expr(:function,
        Expr(:call, :(get_schema), :(trace::$trace_struct_name)),
        Expr(:block,
            :(Gen.StaticAddressSchema(
                Set{Symbol}([$(choice_addrs...)]),
                Set{Symbol}([$(call_addrs...)])))))
end

function generate_trace_type_and_methods(ir::StaticIR, name::Symbol)
    trace_struct_name = gensym("StaticIRTrace_$name")
    trace_struct_expr = generate_trace_struct(ir, trace_struct_name)
    has_choices_expr = generate_has_choices(trace_struct_name)
    get_call_record_expr = generate_get_call_record(ir, trace_struct_name)
    get_assignment_expr = generate_get_assignment(trace_struct_name)
    get_schema_expr = generate_get_schema(ir, trace_struct_name)
    get_leaf_nodes_expr = generate_get_leaf_nodes(ir, trace_struct_name)
    get_internal_nodes_expr = generate_get_internal_nodes(ir, trace_struct_name)
    static_get_leaf_node_exprs = generate_static_get_leaf_node(ir, trace_struct_name)
    static_has_leaf_node_exprs = generate_static_has_leaf_node(ir, trace_struct_name)
    static_get_internal_node_exprs = generate_static_get_internal_node(ir, trace_struct_name)
    static_has_internal_node_exprs = generate_static_has_internal_node(ir, trace_struct_name)
    exprs = Expr(:block, trace_struct_expr, has_choices_expr, get_call_record_expr,
                 get_assignment_expr, get_schema_expr, get_leaf_nodes_expr,
                 get_internal_nodes_expr, static_get_leaf_node_exprs...,
                 static_has_leaf_node_exprs..., static_get_internal_node_exprs...,
                 static_has_internal_node_exprs...)
    (exprs, trace_struct_name)
end
