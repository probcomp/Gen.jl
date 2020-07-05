#__precompile__(false)

module Gen

const generated_functions = []

"""
    load_generated_functions(__module__=Main)

Permit use of generative functions written in the static modeling language up
to this point. Functions are loaded into Main by default.
"""
function load_generated_functions(__module__::Module=Main)
    for function_defn in generated_functions
        Core.eval(__module__, function_defn)
    end
end

"""
    @load_generated_functions

Permit use of generative functions written in the static modeling language up
to this point. Functions are loaded into the calling module.
"""
macro load_generated_functions()
    for function_defn in generated_functions
        Core.eval(__module__, function_defn)
    end
end

export load_generated_functions, @load_generated_functions

# built-in extensions to the reverse mode AD
include("backprop.jl")

# address and address trees
# (including choicemaps and selections)
include("address_tree/address_tree.jl")

# a homogeneous trie data type (not for use as choice map)
include("trie.jl")

# built-in data types for arg-diff and ret-diff values
include("diff.jl")

# generative function interface
include("gen_fn_interface.jl")

# distribution abstract type
include("distribution.jl")

# built-in probability disributions; distribution dsl; combinators
include("modeling_library/modeling_library.jl")

# optimization of trainable parameters
include("optimization.jl")

# dynamic embedded generative function
include("dynamic/dynamic.jl")

# static IR generative function
include("static_ir/static_ir.jl")

# optimization for built-in generative functions (dynamic and static IR)
include("builtin_optimization.jl")

# DSLs for defining dynamic embedded and static IR generative functions
# 'Dynamic DSL' and 'Static DSL'
include("dsl/dsl.jl")

# inference and learning library
include("inference/inference.jl")

end # module Gen
