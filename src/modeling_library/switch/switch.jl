struct Switch{C, N, K, T} <: GenerativeFunction{T, Trace}
    mix::NTuple{N, GenerativeFunction{T}}
    cases::Dict{C, Int}
    function Switch(gen_fns::GenerativeFunction...)
        @assert !isempty(gen_fns)
        rettype = get_return_type(getindex(gen_fns, 1))
        new{Int, length(gen_fns), typeof(gen_fns), rettype}(gen_fns, Dict{Int, Int}())
    end
    function Switch(d::Dict{C, Int}, gen_fns::GenerativeFunction...) where C
        @assert !isempty(gen_fns)
        rettype = get_return_type(getindex(gen_fns, 1))
        new{C, length(gen_fns), typeof(gen_fns), rettype}(gen_fns, d)
    end
end

export Switch

has_argument_grads(switch_fn::Switch) = all(has_argument_grads, switch.mix)
accepts_output_grad(switch_fn::Switch) = all(accepts_output_grad, switch.mix)

function (gen_fn::Switch)(index::Int, args...)
    (_, _, retval) = propose(gen_fn, (index, args...))
    retval
end

function (gen_fn::Switch{C})(index::C, args...) where C
    (_, _, retval) = propose(gen_fn, (gen_fn.cases[index], args...))
    retval
end

include("assess.jl")
include("propose.jl")
include("simulate.jl")
include("generate.jl")
include("update.jl")
include("regenerate.jl")
