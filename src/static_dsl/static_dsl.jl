include("dag_ir.jl")
include("trace.jl")
include("generative_function.jl")

# variable names used in generated code
const trace = gensym("trace")
const weight = gensym("weight")
const subtrace = gensym("subtrace")
const discard = gensym("discard")
const retdiff = gensym("retdiff")

include("generate.jl")
include("update.jl")
