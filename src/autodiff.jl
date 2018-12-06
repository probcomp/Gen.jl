########
# fill #
########

import ReverseDiff

function Base.fill(x::ReverseDiff.TrackedReal{V}, dims::Integer...) where {V}
    tp = ReverseDiff.tape(x)
    out = ReverseDiff.track(fill(ReverseDiff.value(x), dims...), V, tp)
    ReverseDiff.record!(tp, ReverseDiff.SpecialInstruction, fill, (x, dims), out)
    return out
end

@noinline function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(fill)})
    x, dims = instruction.input
    output = instruction.output
    ReverseDiff.istracked(x) && ReverseDiff.increment_deriv!(x, sum(ReverseDiff.deriv(output)))
    ReverseDiff.unseed!(output) 
    return nothing
end 

@noinline function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof(fill)})
    x, dims = instruction.input
    ReverseDiff.value!(instruction.output, fill(value(x), dims...))
    return nothing
end 
