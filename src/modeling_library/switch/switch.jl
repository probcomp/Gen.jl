struct Switch{T1, T2, Tr} <: GenerativeFunction{Union{T1, T2}, Tr}
    a::GenerativeFunction{T1, Tr}
    b::GenerativeFunction{T2, Tr}
end

export Switch

has_argument_grads(switch_fn::Switch) = has_argument_grads(switch_fn.a) && has_argument_grads(switch_fn.b)
accepts_output_grad(switch_fn::Switch) = accepts_output_grad(switch_fn.a) && accepts_output_grad(switch_fn.b)

function (gen_fn::Switch)(flip_p::Float64, args...)
    (_, _, retval) = propose(gen_fn, (flip_p, args...))
    retval
end

include("assess.jl")
include("propose.jl")
include("simulate.jl")
include("generate.jl")
