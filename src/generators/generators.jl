const generated_functions = []

function load_generated_functions()
    for function_defn in generated_functions
        Core.eval(Main, function_defn)
    end
end

# built-in generator types
include("vector_trace.jl")
include("lightweight/lightweight.jl")
include("tree/tree.jl")
include("plate/plate.jl")
include("plate_of_dists/plate_of_dists.jl")
include("markov/markov.jl")
include("basic/basic.jl")
include("at_dynamic/at_dynamic.jl")

export load_generated_functions
