module Gen
    include("address.jl")
    include("choice_trie.jl")
    include("homogenous_trie.jl")
    include("generator.jl")
    include("distribution.jl")
    include("dsl_common.jl")
    include("generators/generators.jl")
    include("injective.jl")
    include("selection.jl")
    include("inference.jl")
end # module
