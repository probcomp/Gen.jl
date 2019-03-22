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
        struct $gen_fn_type_name <: $(QuoteNode(StaticIRGenerativeFunction)){$return_type,$trace_type}
            params_grad::Dict{Symbol,Any}
            params::Dict{Symbol,Any}
        end
        (gen_fn::$gen_fn_type_name)(args...) = propose(gen_fn, args)[3]
        $(Expr(:(.), Gen, QuoteNode(:get_ir)))(::Type{$gen_fn_type_name}) = $(QuoteNode(ir))
        $(Expr(:(.), Gen, QuoteNode(:get_trace_type)))(::Type{$gen_fn_type_name}) = $trace_struct_name
        $(Expr(:(.), Gen, QuoteNode(:has_argument_grads)))(::$gen_fn_type_name) = $(QuoteNode(has_argument_grads))
        $(Expr(:(.), Gen, QuoteNode(:accepts_output_grad)))(::$gen_fn_type_name) = $(QuoteNode(accepts_output_grad))
        $(Expr(:(.), Gen, QuoteNode(:get_gen_fn)))(trace::$trace_struct_name) = $(Expr(:(.), :trace, QuoteNode(static_ir_gen_fn_ref)))
        $(Expr(:(.), Gen, QuoteNode(:get_gen_fn_type)))(::Type{$trace_struct_name}) = $gen_fn_type_name
        $(Expr(:(.), Gen, QuoteNode(:get_options)))(::Type{$gen_fn_type_name}) = $(QuoteNode(options))
    end
    Expr(:block, trace_defns, gen_fn_defn, Expr(:call, gen_fn_type_name, :(Dict{Symbol,Any}()), :(Dict{Symbol,Any}())))
end

include("render_ir.jl")

###########################
# generative function API #
###########################

# variable names used in generated code
const trace = gensym("trace")
const weight = gensym("weight")
const subtrace = gensym("subtrace")

# quoted values and function called in generated code (since generated code is
# evaluted in the user's Main module, not Gen)
const qn_isempty = QuoteNode(isempty)
const qn_get_score = QuoteNode(get_score)
const qn_get_retval = QuoteNode(get_retval)
const qn_project = QuoteNode(project)
const qn_logpdf = QuoteNode(logpdf)
const qn_get_choices = QuoteNode(get_choices)
const qn_random = QuoteNode(random)
const qn_simulate = QuoteNode(simulate)
const qn_generate = QuoteNode(generate)
const qn_update = QuoteNode(update)
const qn_regenerate = QuoteNode(regenerate)
const qn_extend = QuoteNode(extend)
const qn_strip_diff = QuoteNode(strip_diff)
const qn_get_diff = QuoteNode(get_diff)
const qn_Diffed = QuoteNode(Diffed)
const qn_unknown_change = QuoteNode(UnknownChange())
const qn_no_change = QuoteNode(NoChange())
const qn_get_internal_node = QuoteNode(get_internal_node)
const qn_static_get_value = QuoteNode(static_get_value)
const qn_static_get_submap = QuoteNode(static_get_submap)
const qn_static_get_internal_node = QuoteNode(static_get_internal_node)
const qn_empty_choice_map = QuoteNode(EmptyChoiceMap())
const qn_empty_address_set = QuoteNode(EmptyAddressSet())

include("simulate.jl")
include("generate.jl")
include("project.jl")
include("update.jl")
include("backprop.jl")
