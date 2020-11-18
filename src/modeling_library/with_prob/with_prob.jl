struct WithProbability{T} <: GenerativeFunction{T, Trace}
    a::GenerativeFunction{T}
    b::GenerativeFunction{T}
end

export WithProbability

has_argument_grads(switch_fn::WithProbability) = has_argument_grads(switch_fn.a) && has_argument_grads(switch_fn.b)
accepts_output_grad(switch_fn::WithProbability) = accepts_output_grad(switch_fn.a) && accepts_output_grad(switch_fn.b)

function (gen_fn::WithProbability)(flip_p::Float64, args...)
    (_, _, retval) = propose(gen_fn, (flip_p, args...))
    retval
end

include("assess.jl")
include("propose.jl")
include("simulate.jl")
include("generate.jl")
