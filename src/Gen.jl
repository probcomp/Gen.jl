module Gen
    include("address.jl")
    include("assignment.jl")
    include("homogenous_trie.jl")
    include("generative_function.jl")
    include("distribution.jl")
    include("dsl_common.jl")
    include("dynamic_dsl/dynamic_dsl.jl")
    include("static_dsl/static_dsl.jl")
    include("combinators/combinators.jl")
    include("injective.jl")
    include("selection.jl")
    include("inference.jl")
end # module
