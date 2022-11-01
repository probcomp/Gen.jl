#############################
# Directed acyclic graph IR #
#############################

include("dag.jl")


##############################
# generative functions types #
##############################

struct StaticIRGenerativeFunctionOptions
    track_diffs::Bool
    cache_julia_nodes::Bool
end

# trace code generation
include("trace.jl")

"""
    StaticIRGenerativeFunction{T,U} <: GenerativeFunction{T,U}

Abstact type for a static IR generative function with return type T and trace type U.

Contains an intermediate representation based on a directed acyclic graph.
Most generative function interface methods are generated from the intermediate representation.
"""
abstract type StaticIRGenerativeFunction{T,U} <: GenerativeFunction{T,U} end

function get_ir end
function get_gen_fn_type end
function get_options end

function generate_generative_function(ir::StaticIR, name::Symbol; track_diffs=false, cache_julia_nodes=true)
    options = StaticIRGenerativeFunctionOptions(track_diffs, cache_julia_nodes)
    generate_generative_function(ir, name, options)
end

function generate_generative_function(ir::StaticIR, name::Symbol, options::StaticIRGenerativeFunctionOptions)

    (trace_defns, trace_struct_name) = generate_trace_type_and_methods(ir, name, options)

    gen_fn_type_name = gensym("StaticGenFunction_$name")
    return_type = ir.return_node.typ
    trace_type = trace_struct_name
    has_argument_grads = tuple(map((node) -> node.compute_grad, ir.arg_nodes)...)
    accepts_output_grad = ir.accepts_output_grad

    gen_fn_defn = quote
        # Generate type definition
        struct $gen_fn_type_name <: $(QuoteNode(StaticIRGenerativeFunction)){$return_type,$trace_type}
            params_grad::Dict{Symbol,Any}
            params::Dict{Symbol,Any}
        end

        # Generate accessors
        $(GlobalRef(Gen, :get_ir))(::$gen_fn_type_name) = $(QuoteNode(ir))
        $(GlobalRef(Gen, :get_ir))(::Type{$gen_fn_type_name}) = $(QuoteNode(ir))
        $(GlobalRef(Gen, :get_trace_type))(::Type{$gen_fn_type_name}) = $trace_struct_name
        $(GlobalRef(Gen, :has_argument_grads))(::$gen_fn_type_name) = $(QuoteNode(has_argument_grads))
        $(GlobalRef(Gen, :accepts_output_grad))(::$gen_fn_type_name) = $(QuoteNode(accepts_output_grad))
        $(GlobalRef(Gen, :get_gen_fn))(trace::$trace_struct_name) = $(Expr(:(.), :trace, QuoteNode(static_ir_gen_fn_ref)))
        $(GlobalRef(Gen, :get_gen_fn_type))(::Type{$trace_struct_name}) = $gen_fn_type_name
        $(GlobalRef(Gen, :get_options))(::Type{$gen_fn_type_name}) = $(QuoteNode(options))
        # Generate GFI definitions
        (gen_fn::$gen_fn_type_name)(args...) = $(GlobalRef(Gen, :propose))(gen_fn, args)[3]
        @generated function $(GlobalRef(Gen, :simulate))(gen_fn::$gen_fn_type_name, args::$(QuoteNode(Tuple)))
            $(QuoteNode(codegen_simulate))(gen_fn, args)
        end
        @generated function $(GlobalRef(Gen, :generate))(gen_fn::$gen_fn_type_name, args::$(QuoteNode(Tuple)),
                                                         constraints::$(QuoteNode(ChoiceMap)))
            $(QuoteNode(codegen_generate))(gen_fn, args, constraints)
        end
        @generated function $(GlobalRef(Gen, :update))(trace::$trace_type, args::$(QuoteNode(Tuple)),
                                                       argdiffs::$(QuoteNode(Tuple)), constraints::$(QuoteNode(ChoiceMap)))
            $(QuoteNode(codegen_update))(trace, args, argdiffs, constraints)
        end
        @generated function $(GlobalRef(Gen, :regenerate))(trace::$trace_type, args::$(QuoteNode(Tuple)),
                                                           argdiffs::$(QuoteNode(Tuple)), selection::$(QuoteNode(Selection)))
            $(QuoteNode(codegen_regenerate))(trace, args, argdiffs, selection)
        end
        @generated function $(GlobalRef(Gen, :project))(trace::$trace_type, selection::$(QuoteNode(Selection)))
            $(QuoteNode(codegen_project))(trace, selection)
        end
        function $(GlobalRef(Gen, :project))(trace::$trace_type, selection::$(QuoteNode(EmptySelection)))
            trace.$total_noise_fieldname
        end
        @generated function $(GlobalRef(Gen, :choice_gradients))(trace::$trace_type, selection::$(QuoteNode(Selection)),
                                                                 retval_grad)
            $(QuoteNode(codegen_choice_gradients))(trace, selection, retval_grad)
        end
        @generated function $(GlobalRef(Gen, :accumulate_param_gradients!))(trace::$trace_type, retval_grad)
            $(QuoteNode(codegen_accumulate_param_gradients!))(trace, retval_grad)
        end
    end
    Expr(:block, trace_defns, gen_fn_defn, Expr(:call, gen_fn_type_name, :(Dict{Symbol,Any}()), :(Dict{Symbol,Any}())))
end

include("print_ir.jl")
include("render_ir.jl")

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
