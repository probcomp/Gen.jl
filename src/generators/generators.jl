const generated_functions = []

function load_generated_functions()
    #Core.eval(Main, quote
        #import Gen: value_field, is_empty_field, CallRecord
        #import Gen: BasicGenFunction, BasicBlockSimulateState, get_trace_type, get_ir, call_record_field
    #end)
    for function_defn in generated_functions
        Core.eval(Main, function_defn)
    #quote
        #@generated function Gen.generate(gen::Generator, args, constraints, read_trace=nothing)
            #Gen.codegen_generate(gen, args, constraints, read_trace)
        #end
#
        #@generated function Gen.update(gen::Generator, new_args, args_change, trace, constraints, read_trace=nothing)
            #Gen.codegen_update(gen, new_args, args_change, trace, constraints, read_trace)
        #end
#
        #@generated function Gen.project(gen::Generator, args, constraints, read_trace=nothing)
            #Gen.codegen_project(gen, args, constraints, read_trace)
        #end
#
        #@generated function Gen.assess(gen::Generator, args, constraints, read_trace=nothing)
            #Gen.codegen_assess(gen, args, constraints, read_trace)
        #end
#
        #@generated function Gen.simulate(gen::Generator, args, read_trace=nothing)
            #Gen.codegen_simulate(gen, args, read_trace)
        #end
    #end
    end
end

# built-in generator types
include("vector_trace.jl")
include("vector_dist_trace.jl")
include("lightweight/lightweight.jl")
include("plate/plate.jl")
include("plate_of_dists/plate_of_dists.jl")
include("basic/basic.jl")
include("at_dynamic/at_dynamic.jl")

export load_generated_functions
