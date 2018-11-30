#################################
# static IR generative function #
#################################

const generated_functions = []

function load_generated_functions()
    for function_defn in generated_functions
        Core.eval(Main, function_defn)
    end
end
abstract type StaticIRGenerativeFunction{T,U} <: GenerativeFunction{T,U} end
function get_ir end

# trace_exprs = generate_trace_type_and_methods(ir, name)

# TODO add trainable parameters

function generate_generative_function(ir::StaticIR, name::Symbol)

    (trace_defns, trace_struct_name) = generate_trace_type_and_methods(ir, name)

    gen_fn_type_name = gensym("StaticGenFunction_$name")
    return_type = QuoteNode(ir.return_node.typ)
    trace_type = trace_struct_name

    # TODO beautify generated code (remove quote, factor)
    gen_fn_defn = quote
        struct $gen_fn_type_name <: Gen.StaticIRGenerativeFunction{$return_type,$trace_type}
        end
        #(gen_fn::$gen_fn_type_name)(args...) = get_call_record(simulate(gen, args)).retval
        Gen.get_ir(::Type{$gen_fn_type_name}) = $(QuoteNode(ir))
        Gen.get_trace_type(::Type{$gen_fn_type_name}) = $trace_struct_name
        $name = $gen_fn_type_name()
    end
    Expr(:block, trace_defns, gen_fn_defn)
end
