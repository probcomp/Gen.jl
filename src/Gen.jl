#__precompile__(false)

module Gen

"""
    load_generated_functions(__module__=Main)

!!! warning "Deprecation Warning"
    Calling this function is no longer necessary in order to use the static
    modeling language. This function will be removed in a future release.

Previously, this permitted the use of generative functions written in the
static modeling language by loading their associated function definitions into
the specified module.
"""
function load_generated_functions(__module__::Module=Main)
    @warn "`Gen.load_generated_functions` is no longer necessary" *
          " and will be removed in a future release."
end

"""
    @load_generated_functions

!!! warning "Deprecation Warning"
    Calling this macro is no longer necessary in order to use the static
    modeling language. This macro will be removed in a future release.

Previously, this permitted the use of generative functions written in the
static modeling language by loading their associated function definitions into
the calling module.
"""
macro load_generated_functions()
    @warn "`Gen.@load_generated_functions` is no longer necessary" *
          " and will be removed in a future release."
end

export load_generated_functions, @load_generated_functions

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
