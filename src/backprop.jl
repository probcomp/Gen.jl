import ReverseDiff
using ReverseDiff: InstructionTape, SpecialInstruction

# do not track through Bool and Integers
ReverseDiff.track(x::Bool, tape::InstructionTape=InstructionTape()) = x
ReverseDiff.track(x::Integer, tape::InstructionTape=InstructionTape()) = x

# if a value can't be tracked, return the untracked value silently
ReverseDiff.track(x, tape::InstructionTape) = x

########
# fill #
########

function Base.fill(x::ReverseDiff.TrackedReal{V}, dims::Integer...) where {V}
    tp = ReverseDiff.tape(x)
    out = ReverseDiff.track(fill(ReverseDiff.value(x), dims...), V, tp)
    ReverseDiff.record!(tp, SpecialInstruction, fill, (x, dims), out)
    return out
end

@noinline function ReverseDiff.special_reverse_exec!(instruction::SpecialInstruction{typeof(fill)})
    x, dims = instruction.input
    output = instruction.output
    ReverseDiff.istracked(x) && ReverseDiff.increment_deriv!(x, sum(ReverseDiff.deriv(output)))
    ReverseDiff.unseed!(output) 
    return nothing
end 

@noinline function ReverseDiff.special_forward_exec!(instruction::SpecialInstruction{typeof(fill)})
    x, dims = instruction.input
    ReverseDiff.value!(instruction.output, fill(value(x), dims...))
    return nothing
end 
