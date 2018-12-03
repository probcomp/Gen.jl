#############################
# Directed acyclic graph IR #
#############################

include("dag.jl")


#################################################
# generation of specialized trace types from IR #
#################################################

include("trace.jl")


##############################
# generative functions types #
##############################

abstract type StaticIRGenerativeFunction{T,U} <: GenerativeFunction{T,U} end
function get_ir end

# TODO add trainable parameters

function generate_generative_function(ir::StaticIR, name::Symbol)

    (trace_defns, trace_struct_name) = generate_trace_type_and_methods(ir, name)

    gen_fn_type_name = gensym("StaticGenFunction_$name")
    return_type = QuoteNode(ir.return_node.typ)
    trace_type = trace_struct_name
    has_argument_grads = tuple(map((node) -> node.compute_grad, ir.arg_nodes)...)

    gen_fn_defn = quote
        struct $gen_fn_type_name <: Gen.StaticIRGenerativeFunction{$return_type,$trace_type}
        end
        (gen_fn::$gen_fn_type_name)(args...) = get_call_record(simulate(gen_fn, args)).retval
        Gen.get_ir(::Type{$gen_fn_type_name}) = $(QuoteNode(ir))
        Gen.get_trace_type(::Type{$gen_fn_type_name}) = $trace_struct_name
        Gen.has_argument_grads(::$gen_fn_type_name) = $(QuoteNode(has_argument_grads))
        $name = $gen_fn_type_name()
    end
    Expr(:block, trace_defns, gen_fn_defn)
end


###########################
# generative function API #
###########################

# variable names used in generated code
const trace = gensym("trace")
const weight = gensym("weight")
const subtrace = gensym("subtrace")
const discard = gensym("discard")
const retdiff = gensym("retdiff")

include("generate.jl")
include("update.jl")
include("backprop.jl")
