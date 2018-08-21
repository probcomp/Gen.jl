macro load_generated_functions()
    quote
        @generated function GenLite.generate(gen::Generator, args, constraints, read_trace=nothing)
            GenLite.codegen_generate(gen, args, constraints, read_trace)
        end

        @generated function GenLite.update(gen::Generator, new_args, args_change, trace, constraints, read_trace=nothing)#, discard_proto=GenericChoiceTrie())
            GenLite.codegen_update(gen, new_args, args_change, trace, constraints, read_trace, discard_proto)
        end

        @generated function GenLite.project(gen::Generator, args, constraints, read_trace=nothing)
            GenLite.codegen_project(gen, args, constraints, read_trace, discard_proto)
        end

        @generated function GenLite.assess(gen::Generator, args, constraints, read_trace=nothing)
            GenLite.codegen_assess(gen, args, constraints, read_trace)
        end

        @generated function GenLite.simulate(gen::Generator, args, read_trace=nothing)
            GenLite.codegen_simulate(gen, args, read_trace)
        end
    end
end

# built-in generator types
include("vector_trace.jl")
include("lightweight/lightweight.jl")
include("plate/plate.jl")
include("basic/basic.jl")
include("at_dynamic/at_dynamic.jl")

export @load_generated_functions
