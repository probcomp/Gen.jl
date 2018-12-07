module Gen

    const generated_functions = []
    function load_generated_functions()
        for function_defn in generated_functions
            Core.eval(Main, function_defn)
        end
    end
    export generated_functions

    include("autodiff.jl")
    include("address.jl")
    include("assignment.jl")
    include("homogenous_trie.jl")
    include("generative_function_interface.jl")
    include("distribution.jl")
    include("dsl_common.jl")
    include("dynamic_dsl/dynamic_dsl.jl")
    include("static_ir/static_ir.jl")
    include("static_dsl/static_dsl.jl")
    include("combinators/combinators.jl")
    include("injective.jl")
    include("selection.jl")
    include("inference/inference.jl")
end # module
