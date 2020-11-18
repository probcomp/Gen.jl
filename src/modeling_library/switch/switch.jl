struct Switch{N, K, T} <: GenerativeFunction{T, Trace}
    mix::NTuple{N, GenerativeFunction{T}}
    function Switch(gen_fns::GenerativeFunction...)
        @assert !isempty(gen_fns)
        rettype = get_return_type(getindex(gen_fns, 1))
        new{length(gen_fns), typeof(gen_fns), rettype}(gen_fns)
    end
end

export Switch

has_argument_grads(switch_fn::Switch) = all(has_argument_grads, switch.mix)
accepts_output_grad(switch_fn::Switch) = all(accepts_output_grad, switch.mix)

function (gen_fn::Switch)(index::Int, args...)
    (_, _, retval) = propose(gen_fn, (index, args...))
    retval
end

include("assess.jl")
include("propose.jl")
include("simulate.jl")
include("generate.jl")
