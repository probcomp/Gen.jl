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

"""
    StaticIRGenerativeFunction{T,U} <: GenerativeFunction{T,U}

Abstact type for a static IR generative function with return type T and trace type U.

Contains an intermediate representation based on a directed acyclic graph.
Most generative function interface methods are generated from the intermediate representation.
"""
abstract type StaticIRGenerativeFunction{T,U} <: GenerativeFunction{T,U} end

function get_ir end
function get_gen_fn_type end

# TODO add trainable parameters

function generate_generative_function(ir::StaticIR, name::Symbol)

    (trace_defns, trace_struct_name) = generate_trace_type_and_methods(ir, name)

    gen_fn_type_name = gensym("StaticGenFunction_$name")
    return_type = ir.return_node.typ
    trace_type = trace_struct_name
    has_argument_grads = tuple(map((node) -> node.compute_grad, ir.arg_nodes)...)
    accepts_output_grad = ir.accepts_output_grad

    gen_fn_defn = quote
        struct $gen_fn_type_name <: Gen.StaticIRGenerativeFunction{$return_type,$trace_type}
        end
        (gen_fn::$gen_fn_type_name)(args...) = propose(gen_fn, args)[3]
        Gen.get_ir(::Type{$gen_fn_type_name}) = $(QuoteNode(ir))
        Gen.get_trace_type(::Type{$gen_fn_type_name}) = $trace_struct_name
        Gen.has_argument_grads(::$gen_fn_type_name) = $(QuoteNode(has_argument_grads))
        Gen.accepts_output_grad(::$gen_fn_type_name) = $(QuoteNode(accepts_output_grad))
        Gen.get_gen_fn(::$trace_struct_name) = $gen_fn_type_name()
        Gen.get_gen_fn_type(::Type{$trace_struct_name}) = $gen_fn_type_name
    end
    Expr(:block, trace_defns, gen_fn_defn, Expr(:call, gen_fn_type_name))
end


###########################
# generative function API #
###########################

# variable names used in generated code
const trace = gensym("trace")
const weight = gensym("weight")
const subtrace = gensym("subtrace")

include("simulate.jl")
include("generate.jl")
include("project.jl")
include("update.jl")
include("backprop.jl")
