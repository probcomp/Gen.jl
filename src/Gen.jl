module Gen

    const generated_functions = []
    function load_generated_functions()
        for function_defn in generated_functions
            Core.eval(Main, function_defn)
        end
    end
    export generated_functions

    # built-in extensions to the reverse mode AD
    include("backprop.jl")

    # addresses and address selections
    include("address.jl")

    # abstract and built-in concrete choice map data types
    include("choice_map.jl")

    # a homogeneous trie data type (not for use as choice map)
    include("trie.jl")

    # generative function interface
    include("gen_fn_interface.jl")

    # built-in data types for arg-diff and ret-diff values
    include("diff.jl")

    # built-in probability disributions
    include("distribution.jl")

    # utilities for parsing
    include("dsl_common.jl")

    # optimization of trainable parameters
    include("optimization.jl")

    # dynamic embedded generative function
    include("dynamic/dynamic.jl")

    # static IR generative function
    include("static_ir/static_ir.jl")

    # DSLs for defining dynamic embedded and static IR generative functions
    # 'Dynamic DSL' and 'Static DSL'
    include("dsl/dsl.jl")

    # generative function combinators
    include("combinators/combinators.jl")

    # injective function DSL (not currently documented)
    include("injective.jl")

    # selection DSL (not currently documented)
    include("selection.jl")

    # inference and learning library
    include("inference/inference.jl")

end # module Gen
